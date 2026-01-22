import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _metadata_for_project(self, project: Dict) -> Dict:
    """Gets project metadata for all files"""
    project_id = project.get(ID_KEY)
    url = f'{self.api}/projects/{project_id}/artifacts/latest'
    all_artifacts = []
    per_file_metadata: Dict = {}
    while url:
        response = requests.request('GET', url, headers={'Authorization': f'Bearer {self.access_token}'}, data={})
        if response.ok:
            data = response.json()
            all_artifacts.extend(data['artifacts'])
            url = data.get('next', None)
        elif response.status_code == 404:
            return per_file_metadata
        else:
            raise Exception(f'Failed to download {url} (status: {response.status_code})')
    for artifact in all_artifacts:
        artifact_name = artifact.get('name')
        artifact_url = artifact.get('url')
        artifact_doc = artifact.get('document')
        if artifact_name == 'report-values.xml' and artifact_url and artifact_doc:
            doc_id = artifact_doc[ID_KEY]
            metadata: Dict = {}
            response = requests.request('GET', f'{artifact_url}/content', headers={'Authorization': f'Bearer {self.access_token}'}, data={})
            if response.ok:
                try:
                    from lxml import etree
                except ImportError:
                    raise ImportError('Could not import lxml python package. Please install it with `pip install lxml`.')
                artifact_tree = etree.parse(io.BytesIO(response.content))
                artifact_root = artifact_tree.getroot()
                ns = artifact_root.nsmap
                entries = artifact_root.xpath('//pr:Entry', namespaces=ns)
                for entry in entries:
                    heading = entry.xpath('./pr:Heading', namespaces=ns)[0].text
                    value = ' '.join(entry.xpath('./pr:Value', namespaces=ns)[0].itertext()).strip()
                    metadata[heading] = value[:self.max_metadata_length]
                per_file_metadata[doc_id] = metadata
            else:
                raise Exception(f'Failed to download {artifact_url}/content ' + '(status: {response.status_code})')
    return per_file_metadata