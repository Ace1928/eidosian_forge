from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Type
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
class BaseBrowserTool(BaseTool):
    """Base class for browser tools."""
    sync_browser: Optional['SyncBrowser'] = None
    async_browser: Optional['AsyncBrowser'] = None

    @root_validator
    def validate_browser_provided(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        lazy_import_playwright_browsers()
        if values.get('async_browser') is None and values.get('sync_browser') is None:
            raise ValueError('Either async_browser or sync_browser must be specified.')
        return values

    @classmethod
    def from_browser(cls, sync_browser: Optional[SyncBrowser]=None, async_browser: Optional[AsyncBrowser]=None) -> BaseBrowserTool:
        """Instantiate the tool."""
        lazy_import_playwright_browsers()
        return cls(sync_browser=sync_browser, async_browser=async_browser)