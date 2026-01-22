from __future__ import annotations
import logging
from typing import (
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
class _MinimaxEndpointClient(BaseModel):
    """An API client that talks to a Minimax llm endpoint."""
    host: str
    group_id: str
    api_key: SecretStr
    api_url: str

    @root_validator(pre=True, allow_reuse=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if 'api_url' not in values:
            host = values['host']
            group_id = values['group_id']
            api_url = f'{host}/v1/text/chatcompletion?GroupId={group_id}'
            values['api_url'] = api_url
        return values

    def post(self, request: Any) -> Any:
        headers = {'Authorization': f'Bearer {self.api_key.get_secret_value()}'}
        response = requests.post(self.api_url, headers=headers, json=request)
        if not response.ok:
            raise ValueError(f'HTTP {response.status_code} error: {response.text}')
        if response.json()['base_resp']['status_code'] > 0:
            raise ValueError(f'API {response.json()['base_resp']['status_code']} error: {response.json()['base_resp']['status_msg']}')
        return response.json()['reply']