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
def apply_and_parse(self, input_list: List[Dict[str, Any]], callbacks: Callbacks=None) -> Sequence[Union[str, List[str], Dict[str, str]]]:
    """Call apply and then parse the results."""
    warnings.warn('The apply_and_parse method is deprecated, instead pass an output parser directly to LLMChain.')
    result = self.apply(input_list, callbacks=callbacks)
    return self._parse_generation(result)