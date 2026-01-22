import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _load_sheet_from_id(self, id: str) -> List[Document]:
    """Load a sheet and all tabs from an ID."""
    from googleapiclient.discovery import build
    creds = self._load_credentials()
    sheets_service = build('sheets', 'v4', credentials=creds)
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=id).execute()
    sheets = spreadsheet.get('sheets', [])
    documents = []
    for sheet in sheets:
        sheet_name = sheet['properties']['title']
        result = sheets_service.spreadsheets().values().get(spreadsheetId=id, range=sheet_name).execute()
        values = result.get('values', [])
        if not values:
            continue
        header = values[0]
        for i, row in enumerate(values[1:], start=1):
            metadata = {'source': f'https://docs.google.com/spreadsheets/d/{id}/edit?gid={sheet['properties']['sheetId']}', 'title': f'{spreadsheet['properties']['title']} - {sheet_name}', 'row': i}
            content = []
            for j, v in enumerate(row):
                title = header[j].strip() if len(header) > j else ''
                content.append(f'{title}: {v.strip()}')
            page_content = '\n'.join(content)
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents