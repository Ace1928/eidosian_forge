import json
import urllib.request
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.utils import stringify_dict
from langchain_community.document_loaders.base import BaseLoader
def _get_figma_file(self) -> Any:
    """Get Figma file from Figma REST API."""
    headers = {'X-Figma-Token': self.access_token}
    request = urllib.request.Request(self._construct_figma_api_url(), headers=headers)
    with urllib.request.urlopen(request) as response:
        json_data = json.loads(response.read().decode())
        return json_data