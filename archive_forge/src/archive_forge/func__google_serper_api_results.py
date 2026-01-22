from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from typing_extensions import Literal
def _google_serper_api_results(self, search_term: str, search_type: str='search', **kwargs: Any) -> dict:
    headers = {'X-API-KEY': self.serper_api_key or '', 'Content-Type': 'application/json'}
    params = {'q': search_term, **{key: value for key, value in kwargs.items() if value is not None}}
    response = requests.post(f'https://google.serper.dev/{search_type}', headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results