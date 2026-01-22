from typing import Optional
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.searx_search import SearxSearchWrapper
class SearxSearchRun(BaseTool):
    """Tool that queries a Searx instance."""
    name: str = 'searx_search'
    description: str = 'A meta search engine.Useful for when you need to answer questions about current events.Input should be a search query.'
    wrapper: SearxSearchWrapper
    kwargs: dict = Field(default_factory=dict)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return self.wrapper.run(query, **self.kwargs)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the tool asynchronously."""
        return await self.wrapper.arun(query, **self.kwargs)