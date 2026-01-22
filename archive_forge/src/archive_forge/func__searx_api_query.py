import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import (
from langchain_core.utils import get_from_dict_or_env
def _searx_api_query(self, params: dict) -> SearxResults:
    """Actual request to searx API."""
    raw_result = requests.get(self.searx_host, headers=self.headers, params=params, verify=not self.unsecure)
    if not raw_result.ok:
        raise ValueError('Searx API returned an error: ', raw_result.text)
    res = SearxResults(raw_result.text)
    self._result = res
    return res