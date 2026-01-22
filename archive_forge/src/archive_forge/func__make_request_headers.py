from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
def _make_request_headers(self, headers: Optional[Dict]=None) -> Dict:
    headers = headers or {}
    if not isinstance(self.arcee_api_key, SecretStr):
        raise TypeError(f'arcee_api_key must be a SecretStr. Got {type(self.arcee_api_key)}')
    api_key = self.arcee_api_key.get_secret_value()
    internal_headers = {'X-Token': api_key, 'Content-Type': 'application/json'}
    headers.update(internal_headers)
    return headers