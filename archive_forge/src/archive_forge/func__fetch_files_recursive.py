import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _fetch_files_recursive(self, service: Any, folder_id: str) -> List[Dict[str, Union[str, List[str]]]]:
    """Fetch all files and subfolders recursively."""
    results = service.files().list(q=f"'{folder_id}' in parents", pageSize=1000, includeItemsFromAllDrives=True, supportsAllDrives=True, fields='nextPageToken, files(id, name, mimeType, parents, trashed)').execute()
    files = results.get('files', [])
    returns = []
    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            if self.recursive:
                returns.extend(self._fetch_files_recursive(service, file['id']))
        else:
            returns.append(file)
    return returns