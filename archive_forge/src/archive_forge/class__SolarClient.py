from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
class _SolarClient(BaseModel):
    """An API client that talks to the Solar server."""
    api_key: SecretStr
    'The API key to use for authentication.'
    base_url: str = SOLAR_SERVICE_URL_BASE

    def completion(self, request: Any) -> Any:
        headers = {'Authorization': f'Bearer {self.api_key.get_secret_value()}'}
        response = requests.post(f'{self.base_url}/chat/completions', headers=headers, json=request)
        if not response.ok:
            raise ValueError(f'HTTP {response.status_code} error: {response.text}')
        return response.json()['choices'][0]['message']['content']