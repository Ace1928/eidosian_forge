import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _get_folder(self, folder_id: str) -> str:
    req_folder = urllib.request.Request(self._get_folder_url.format(id=folder_id))
    with urllib.request.urlopen(req_folder) as response:
        json_data = json.loads(response.read().decode())
        return json_data['title']