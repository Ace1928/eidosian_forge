import logging
from typing import Any, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_diffbot_data(self, url: str) -> Any:
    """Get Diffbot file from Diffbot REST API."""
    diffbot_url = self._diffbot_api_url('article')
    params = {'token': self.api_token, 'url': url}
    response = requests.get(diffbot_url, params=params, timeout=10)
    return response.json() if response.ok else {}