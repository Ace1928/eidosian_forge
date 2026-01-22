from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models import (
from langchain_core.load.dump import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.runnables import (
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from langchain.chains.base import Chain
def _get_num_tokens(self, text: str) -> int:
    return _get_language_model(self.llm).get_num_tokens(text)