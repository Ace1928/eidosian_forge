import warnings
from typing import Any, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
def DuckDuckGoSearchTool(*args: Any, **kwargs: Any) -> DuckDuckGoSearchRun:
    """
    Deprecated. Use DuckDuckGoSearchRun instead.

    Args:
        *args:
        **kwargs:

    Returns:
        DuckDuckGoSearchRun
    """
    warnings.warn('DuckDuckGoSearchTool will be deprecated in the future. Please use DuckDuckGoSearchRun instead.', DeprecationWarning)
    return DuckDuckGoSearchRun(*args, **kwargs)