from __future__ import annotations
import logging
from typing import (
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
class MinimaxCommon(BaseModel):
    """Common parameters for Minimax large language models."""
    _client: _MinimaxEndpointClient
    model: str = 'abab5.5-chat'
    'Model name to use.'
    max_tokens: int = 256
    'Denotes the number of tokens to predict per generation.'
    temperature: float = 0.7
    'A non-negative float that tunes the degree of randomness in generation.'
    top_p: float = 0.95
    'Total probability mass of tokens to consider at each step.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Holds any model parameters valid for `create` call not explicitly specified.'
    minimax_api_host: Optional[str] = None
    minimax_group_id: Optional[str] = None
    minimax_api_key: Optional[SecretStr] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['minimax_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'minimax_api_key', 'MINIMAX_API_KEY'))
        values['minimax_group_id'] = get_from_dict_or_env(values, 'minimax_group_id', 'MINIMAX_GROUP_ID')
        values['minimax_api_host'] = get_from_dict_or_env(values, 'minimax_api_host', 'MINIMAX_API_HOST', default='https://api.minimax.chat')
        values['_client'] = _MinimaxEndpointClient(host=values['minimax_api_host'], api_key=values['minimax_api_key'], group_id=values['minimax_group_id'])
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {'model': self.model, 'tokens_to_generate': self.max_tokens, 'temperature': self.temperature, 'top_p': self.top_p, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{'model': self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'minimax'