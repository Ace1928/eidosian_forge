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
@validator('content_formatter')
def validate_content_formatter(cls, field_value: Any, values: Dict) -> ContentFormatterBase:
    """Validate that content formatter is supported by endpoint type."""
    endpoint_api_type = values.get('endpoint_api_type')
    if endpoint_api_type not in field_value.supported_api_types:
        raise ValueError(f'Content formatter f{type(field_value)} is not supported by this endpoint. Supported types are {field_value.supported_api_types} but endpoint is {endpoint_api_type}.')
    return field_value