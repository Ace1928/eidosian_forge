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
class HFContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the HuggingFace catalog."""

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.dedicated]

    def format_request_payload(self, prompt: str, model_kwargs: Dict, api_type: AzureMLEndpointApiType) -> bytes:
        ContentFormatterBase.escape_special_characters(prompt)
        request_payload = json.dumps({'inputs': [f'"{prompt}"'], 'parameters': model_kwargs})
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes, api_type: AzureMLEndpointApiType) -> Generation:
        try:
            choice = json.loads(output)[0]['0']['generated_text']
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
        return Generation(text=choice)