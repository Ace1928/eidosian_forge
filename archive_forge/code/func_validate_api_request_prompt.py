from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
@root_validator(pre=True)
def validate_api_request_prompt(cls, values: Dict) -> Dict:
    """Check that api request prompt expects the right variables."""
    input_vars = values['api_request_chain'].prompt.input_variables
    expected_vars = {'question', 'api_docs'}
    if set(input_vars) != expected_vars:
        raise ValueError(f'Input variables should be {expected_vars}, got {input_vars}')
    return values