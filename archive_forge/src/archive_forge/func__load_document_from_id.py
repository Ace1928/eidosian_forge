import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _load_document_from_id(self, id: str) -> Document:
    """Load a document from an ID."""
    from io import BytesIO
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    creds = self._load_credentials()
    service = build('drive', 'v3', credentials=creds)
    file = service.files().get(fileId=id, supportsAllDrives=True, fields='modifiedTime,name').execute()
    request = service.files().export_media(fileId=id, mimeType='text/plain')
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    try:
        while done is False:
            status, done = downloader.next_chunk()
    except HttpError as e:
        if e.resp.status == 404:
            print('File not found: {}'.format(id))
        else:
            print('An error occurred: {}'.format(e))
    text = fh.getvalue().decode('utf-8')
    metadata = {'source': f'https://docs.google.com/document/d/{id}/edit', 'title': f'{file.get('name')}', 'when': f'{file.get('modifiedTime')}'}
    return Document(page_content=text, metadata=metadata)