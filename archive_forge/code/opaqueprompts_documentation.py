import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
Return type of LLM.

        This is an override of the base class method.
        