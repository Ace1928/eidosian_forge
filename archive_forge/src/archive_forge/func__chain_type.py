from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.prompt import (
from langchain.chains.sequential import SequentialChain
@property
def _chain_type(self) -> str:
    return 'llm_checker_chain'