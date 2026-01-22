from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
class RedditSearchRun(BaseTool):
    """Tool that queries for posts on a subreddit."""
    name: str = 'reddit_search'
    description: str = 'A tool that searches for posts on Reddit.Useful when you need to know post information on a subreddit.'
    api_wrapper: RedditSearchAPIWrapper = Field(default_factory=RedditSearchAPIWrapper)
    args_schema: Type[BaseModel] = RedditSearchSchema

    def _run(self, query: str, sort: str, time_filter: str, subreddit: str, limit: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query=query, sort=sort, time_filter=time_filter, subreddit=subreddit, limit=int(limit))