from __future__ import annotations
from typing import Any, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.openapi.prompt import DESCRIPTION
from langchain_community.tools import BaseTool
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.tools.requests.tool import (
from langchain_community.utilities.requests import TextRequestsWrapper
class RequestsToolkit(BaseToolkit):
    """Toolkit for making REST requests.

    *Security Note*: This toolkit contains tools to make GET, POST, PATCH, PUT,
        and DELETE requests to an API.

        Exercise care in who is allowed to use this toolkit. If exposing
        to end users, consider that users will be able to make arbitrary
        requests on behalf of the server hosting the code. For example,
        users could ask the server to make a request to a private API
        that is only accessible from the server.

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.
    """
    requests_wrapper: TextRequestsWrapper
    allow_dangerous_requests: bool = False
    'Allow dangerous requests. See documentation for details.'

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [RequestsGetTool(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests), RequestsPostTool(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests), RequestsPatchTool(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests), RequestsPutTool(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests), RequestsDeleteTool(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests)]