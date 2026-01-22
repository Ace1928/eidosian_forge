import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _get_tags(self, note_id: str) -> List[str]:
    req_tag = urllib.request.Request(self._get_tag_url.format(id=note_id))
    with urllib.request.urlopen(req_tag) as response:
        json_data = json.loads(response.read().decode())
        return [tag['title'] for tag in json_data['items']]