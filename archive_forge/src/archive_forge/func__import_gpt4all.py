import warnings
from typing import Any, Callable, Dict, Type
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.llms import BaseLLM
from langchain.utils.interactive_env import is_interactive_env
def _import_gpt4all() -> Any:
    from langchain_community.llms.gpt4all import GPT4All
    return GPT4All