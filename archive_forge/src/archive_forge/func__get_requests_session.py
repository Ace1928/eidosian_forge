from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
def _get_requests_session(self) -> requests.Session:
    """Get a requests session with the correct headers."""
    retry_strategy: Retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter: HTTPAdapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount('https://', adapter)
    session.headers.update({'Authorization': f'App {self.infobip_api_key}', 'User-Agent': 'infobip-langchain-community'})
    return session