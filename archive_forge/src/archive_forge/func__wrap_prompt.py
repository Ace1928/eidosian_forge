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
def _wrap_prompt(self, prompt: str) -> str:
    if not self.HUMAN_PROMPT or not self.AI_PROMPT:
        raise NameError('Please ensure the anthropic package is loaded')
    if prompt.startswith(self.HUMAN_PROMPT):
        return prompt
    corrected_prompt, n_subs = re.subn('^\\n*Human:', self.HUMAN_PROMPT, prompt)
    if n_subs == 1:
        return corrected_prompt
    return f'{self.HUMAN_PROMPT} {prompt}{self.AI_PROMPT} Sure, here you go:\n'