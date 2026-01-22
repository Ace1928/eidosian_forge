import re
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str
def _get_anthropic_stop(self, stop: Optional[List[str]]=None) -> List[str]:
    if not self.HUMAN_PROMPT or not self.AI_PROMPT:
        raise NameError('Please ensure the anthropic package is loaded')
    if stop is None:
        stop = []
    stop.extend([self.HUMAN_PROMPT])
    return stop