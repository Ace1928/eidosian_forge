from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra
from requests import Response
def _get_resp_content(self, response: Response) -> Union[str, Dict[str, Any]]:
    if self.response_content_type == 'text':
        return response.text
    elif self.response_content_type == 'json':
        return response.json()
    else:
        raise ValueError(f'Invalid return type: {self.response_content_type}')