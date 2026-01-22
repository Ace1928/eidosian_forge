import json
from typing import Any, Dict, Optional, Union
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
from langchain_community.utilities.requests import GenericRequestsWrapper
from langchain_core.tools import BaseTool
class RequestsDeleteTool(BaseRequestsTool, BaseTool):
    """Tool for making a DELETE request to an API endpoint."""
    name: str = 'requests_delete'
    description: str = 'A portal to the internet.\n    Use this when you need to make a DELETE request to a URL.\n    Input should be a specific url, and the output will be the text\n    response of the DELETE request.\n    '

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        return self.requests_wrapper.delete(_clean_url(url))

    async def _arun(self, url: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        return await self.requests_wrapper.adelete(_clean_url(url))