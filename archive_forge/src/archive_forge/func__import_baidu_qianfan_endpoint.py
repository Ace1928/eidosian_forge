import warnings
from typing import Any, Callable, Dict, Type
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.llms import BaseLLM
from langchain.utils.interactive_env import is_interactive_env
def _import_baidu_qianfan_endpoint() -> Any:
    from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
    return QianfanLLMEndpoint