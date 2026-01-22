from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
def _transform_prompt(self, prompt: str) -> str:
    """Transform prompt."""
    if self.inject_instruction_format:
        prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
    return prompt