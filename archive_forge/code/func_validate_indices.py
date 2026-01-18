from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.elasticsearch_database.prompts import ANSWER_PROMPT, DSL_PROMPT
from langchain.chains.llm import LLMChain
@root_validator()
def validate_indices(cls, values: dict) -> dict:
    if values['include_indices'] and values['ignore_indices']:
        raise ValueError("Cannot specify both 'include_indices' and 'ignore_indices'.")
    return values