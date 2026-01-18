from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_community.graphs import BaseNeptuneGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
def use_simple_prompt(llm: BaseLanguageModel) -> bool:
    """Decides whether to use the simple prompt"""
    if llm._llm_type and 'anthropic' in llm._llm_type:
        return True
    if hasattr(llm, 'model_id') and 'anthropic' in llm.model_id:
        return True
    return False