import os
import tempfile
import urllib.parse
from typing import Any, List, Optional
from urllib.parse import urljoin
import requests
from langchain_core.documents import Document
from requests.auth import HTTPBasicAuth
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
class LakeFSClient:
    """Client for lakeFS."""

    def __init__(self, lakefs_access_key: str, lakefs_secret_key: str, lakefs_endpoint: str):
        self.__endpoint = '/'.join([lakefs_endpoint, 'api', 'v1/'])
        self.__auth = HTTPBasicAuth(lakefs_access_key, lakefs_secret_key)
        try:
            health_check = requests.get(urljoin(self.__endpoint, 'healthcheck'), auth=self.__auth)
            health_check.raise_for_status()
        except Exception:
            raise ValueError("lakeFS server isn't accessible. Make sure lakeFS is running.")

    def ls_objects(self, repo: str, ref: str, path: str, presign: Optional[bool]) -> List:
        qp = {'prefix': path, 'presign': presign}
        eqp = urllib.parse.urlencode(qp)
        objects_ls_endpoint = urljoin(self.__endpoint, f'repositories/{repo}/refs/{ref}/objects/ls?{eqp}')
        olsr = requests.get(objects_ls_endpoint, auth=self.__auth)
        olsr.raise_for_status()
        olsr_json = olsr.json()
        return list(map(lambda res: (res['path'], res['physical_address']), olsr_json['results']))

    def is_presign_supported(self) -> bool:
        config_endpoint = self.__endpoint + 'config'
        response = requests.get(config_endpoint, auth=self.__auth)
        response.raise_for_status()
        config = response.json()
        return config['storage_config']['pre_sign_support']