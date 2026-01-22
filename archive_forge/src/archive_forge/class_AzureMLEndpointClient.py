import json
import urllib.request
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator, validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class AzureMLEndpointClient(object):
    """AzureML Managed Endpoint client."""

    def __init__(self, endpoint_url: str, endpoint_api_key: str, deployment_name: str='', timeout: int=DEFAULT_TIMEOUT) -> None:
        """Initialize the class."""
        if not endpoint_api_key or not endpoint_url:
            raise ValueError('A key/token and REST endpoint should \n                be provided to invoke the endpoint')
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name
        self.timeout = timeout

    def call(self, body: bytes, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> bytes:
        """call."""
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.endpoint_api_key}
        if self.deployment_name != '':
            headers['azureml-model-deployment'] = self.deployment_name
        req = urllib.request.Request(self.endpoint_url, body, headers)
        response = urllib.request.urlopen(req, timeout=kwargs.get('timeout', self.timeout))
        result = response.read()
        return result