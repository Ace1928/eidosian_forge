from __future__ import annotations
import json
from typing import Any, Dict, List, NamedTuple, Optional, cast
from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.utilities.requests import Requests
from langchain_core.callbacks import CallbackManagerForChainRun, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from requests import Response
from langchain.chains.api.openapi.requests_chain import APIRequesterChain
from langchain.chains.api.openapi.response_chain import APIResponderChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
def _extract_body_params(self, args: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Extract the request body params from the deserialized input."""
    body_params = None
    if self.param_mapping.body_params:
        body_params = {}
        for param in self.param_mapping.body_params:
            if param in args:
                body_params[param] = args.pop(param)
    return body_params