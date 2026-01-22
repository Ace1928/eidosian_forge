from typing import Any, Callable, Dict, Type
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM
def _import_friendli() -> Type[BaseLLM]:
    from langchain_community.llms.friendli import Friendli
    return Friendli