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
class QuestionGeneratorChain(LLMChain):
    """Chain that generates questions from uncertain spans."""
    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT
    'Prompt template for the chain.'

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain."""
        return ['user_input', 'context', 'response']