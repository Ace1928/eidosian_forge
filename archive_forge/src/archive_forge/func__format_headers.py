import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def _format_headers(self) -> Dict[str, str]:
    """Format headers for requests."""
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    if self.zapier_nla_oauth_access_token:
        headers.update({'Authorization': f'Bearer {self.zapier_nla_oauth_access_token}'})
    else:
        headers.update({'X-API-Key': self.zapier_nla_api_key})
    return headers