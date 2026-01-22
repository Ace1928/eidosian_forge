from __future__ import annotations
from typing import Optional, Type
from urllib.parse import urlparse
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
class NavigateToolInput(BaseModel):
    """Input for NavigateToolInput."""
    url: str = Field(..., description='url to navigate to')

    @validator('url')
    def validate_url_scheme(cls, url: str) -> str:
        """Check that the URL scheme is valid."""
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ('http', 'https'):
            raise ValueError("URL scheme must be 'http' or 'https'")
        return url