from __future__ import annotations
import json
from typing import Dict, List, Optional
import aiohttp
import requests
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env, get_from_env
def _build_search_url(self, query: str) -> str:
    url_suffix = get_from_env('', 'AZURE_AI_SEARCH_URL_SUFFIX', DEFAULT_URL_SUFFIX)
    if url_suffix in self.service_name and 'https://' in self.service_name:
        base_url = f'{self.service_name}/'
    elif url_suffix in self.service_name and 'https://' not in self.service_name:
        base_url = f'https://{self.service_name}/'
    elif url_suffix not in self.service_name and 'https://' in self.service_name:
        base_url = f'{self.service_name}.{url_suffix}/'
    elif url_suffix not in self.service_name and 'https://' not in self.service_name:
        base_url = f'https://{self.service_name}.{url_suffix}/'
    else:
        base_url = self.service_name
    endpoint_path = f'indexes/{self.index_name}/docs?api-version={self.api_version}'
    top_param = f'&$top={self.top_k}' if self.top_k else ''
    return base_url + endpoint_path + f'&search={query}' + top_param