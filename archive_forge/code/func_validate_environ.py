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
@root_validator(pre=True)
def validate_environ(cls, values: Dict) -> Dict:
    values['endpoint_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'endpoint_api_key', 'AZUREML_ENDPOINT_API_KEY'))
    values['endpoint_url'] = get_from_dict_or_env(values, 'endpoint_url', 'AZUREML_ENDPOINT_URL')
    values['deployment_name'] = get_from_dict_or_env(values, 'deployment_name', 'AZUREML_DEPLOYMENT_NAME', '')
    values['endpoint_api_type'] = get_from_dict_or_env(values, 'endpoint_api_type', 'AZUREML_ENDPOINT_API_TYPE', AzureMLEndpointApiType.dedicated)
    values['timeout'] = get_from_dict_or_env(values, 'timeout', 'AZUREML_TIMEOUT', str(DEFAULT_TIMEOUT))
    return values