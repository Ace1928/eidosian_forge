from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.adapters import HTTPAdapter, Retry
from typing_extensions import NotRequired, TypedDict
def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings using the Embaas API."""
    payload = self._generate_payload(texts)
    try:
        return self._handle_request(payload)
    except requests.exceptions.RequestException as e:
        if e.response is None or not e.response.text:
            raise ValueError(f'Error raised by embaas embeddings API: {e}')
        parsed_response = e.response.json()
        if 'message' in parsed_response:
            raise ValueError(f'Validation Error raised by embaas embeddings API:{parsed_response['message']}')
        raise