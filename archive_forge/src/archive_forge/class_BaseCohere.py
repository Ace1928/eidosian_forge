from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
from langchain_community.llms.utils import enforce_stop_tokens
@deprecated(since='0.0.30', removal='0.2.0', alternative_import='langchain_cohere.BaseCohere')
class BaseCohere(Serializable):
    """Base class for Cohere models."""
    client: Any
    async_client: Any
    model: Optional[str] = Field(default=None)
    'Model name to use.'
    temperature: float = 0.75
    'A non-negative float that tunes the degree of randomness in generation.'
    cohere_api_key: Optional[SecretStr] = None
    'Cohere API key. If not provided, will be read from the environment variable.'
    stop: Optional[List[str]] = None
    streaming: bool = Field(default=False)
    'Whether to stream the results.'
    user_agent: str = 'langchain'
    'Identifier for the application making the request.'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import cohere
        except ImportError:
            raise ImportError('Could not import cohere python package. Please install it with `pip install cohere`.')
        else:
            values['cohere_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'cohere_api_key', 'COHERE_API_KEY'))
            client_name = values['user_agent']
            values['client'] = cohere.Client(api_key=values['cohere_api_key'].get_secret_value(), client_name=client_name)
            values['async_client'] = cohere.AsyncClient(api_key=values['cohere_api_key'].get_secret_value(), client_name=client_name)
        return values