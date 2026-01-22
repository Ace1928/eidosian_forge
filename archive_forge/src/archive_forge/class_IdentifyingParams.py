from __future__ import annotations
import copy
import json
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import PrivateAttr
class IdentifyingParams(TypedDict):
    """Parameters for identifying a model as a typed dict."""
    model_name: str
    model_id: Optional[str]
    server_url: Optional[str]
    server_type: Optional[ServerType]
    embedded: bool
    llm_kwargs: Dict[str, Any]