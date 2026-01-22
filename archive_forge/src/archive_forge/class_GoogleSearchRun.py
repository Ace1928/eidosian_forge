from typing import Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
@deprecated(since='0.0.33', removal='0.2.0', alternative_import='langchain_google_community.GoogleSearchRun')
class GoogleSearchRun(BaseTool):
    """Tool that queries the Google search API."""
    name: str = 'google_search'
    description: str = 'A wrapper around Google Search. Useful for when you need to answer questions about current events. Input should be a search query.'
    api_wrapper: GoogleSearchAPIWrapper

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)