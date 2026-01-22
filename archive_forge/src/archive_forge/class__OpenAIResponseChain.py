from __future__ import annotations
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
from langchain.chains.llm import LLMChain
class _OpenAIResponseChain(_ResponseChain):
    """Chain that generates responses from user input and context."""
    llm: OpenAI = Field(default_factory=lambda: OpenAI(max_tokens=32, model_kwargs={'logprobs': 1}, temperature=0))

    def _extract_tokens_and_log_probs(self, generations: List[Generation]) -> Tuple[Sequence[str], Sequence[float]]:
        tokens = []
        log_probs = []
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError
            tokens.extend(gen.generation_info['logprobs']['tokens'])
            log_probs.extend(gen.generation_info['logprobs']['token_logprobs'])
        return (tokens, log_probs)