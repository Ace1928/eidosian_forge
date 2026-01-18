import io
import json
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
@abstractmethod
def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
    """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """