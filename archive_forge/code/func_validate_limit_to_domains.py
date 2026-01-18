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
def validate_limit_to_domains(cls, values: Dict) -> Dict:
    """Check that allowed domains are valid."""
    if 'limit_to_domains' not in values:
        raise ValueError('You must specify a list of domains to limit access using `limit_to_domains`')
    if not values['limit_to_domains'] and values['limit_to_domains'] is not None:
        raise ValueError('Please provide a list of domains to limit access using `limit_to_domains`.')
    return values