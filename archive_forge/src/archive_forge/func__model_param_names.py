import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Set
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
@staticmethod
def _model_param_names() -> Set[str]:
    return {'max_tokens', 'temp', 'top_k', 'top_p', 'do_sample'}