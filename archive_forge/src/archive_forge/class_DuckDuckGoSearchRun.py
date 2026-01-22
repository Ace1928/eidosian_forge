import warnings
from typing import Any, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
class DuckDuckGoSearchRun(BaseTool):
    """Tool that queries the DuckDuckGo search API."""
    name: str = 'duckduckgo_search'
    description: str = 'A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.'
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(default_factory=DuckDuckGoSearchAPIWrapper)
    args_schema: Type[BaseModel] = DDGInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)