import json
from typing import Any, Dict, Optional, Union
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
from langchain_community.utilities.requests import GenericRequestsWrapper
from langchain_core.tools import BaseTool
class BaseRequestsTool(BaseModel):
    """Base class for requests tools."""
    requests_wrapper: GenericRequestsWrapper
    allow_dangerous_requests: bool = False

    def __init__(self, **kwargs: Any):
        """Initialize the tool."""
        if not kwargs.get('allow_dangerous_requests', False):
            raise ValueError("You must set allow_dangerous_requests to True to use this tool. Requests can be dangerous and can lead to security vulnerabilities. For example, users can ask a server to make a request to an internal server. It's recommended to use requests through a proxy server and avoid accepting inputs from untrusted sources without proper sandboxing.Please see: https://python.langchain.com/docs/security for further security information.")
        super().__init__(**kwargs)