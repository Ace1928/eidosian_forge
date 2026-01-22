import json
import urllib.request
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.utils import stringify_dict
from langchain_community.document_loaders.base import BaseLoader
def _construct_figma_api_url(self) -> str:
    api_url = 'https://api.figma.com/v1/files/%s/nodes?ids=%s' % (self.key, self.ids)
    return api_url