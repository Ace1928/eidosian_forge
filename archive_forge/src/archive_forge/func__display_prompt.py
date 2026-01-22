from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
def _display_prompt(prompt: str) -> None:
    """Displays the given prompt to the user."""
    print(f'\n{prompt}')