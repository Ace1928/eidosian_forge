from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Type, cast
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.playwright.base import (
from langchain_community.tools.playwright.click import ClickTool
from langchain_community.tools.playwright.current_page import CurrentWebPageTool
from langchain_community.tools.playwright.extract_hyperlinks import (
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.tools.playwright.get_elements import GetElementsTool
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.navigate_back import NavigateBackTool
@root_validator
def validate_imports_and_browser_provided(cls, values: dict) -> dict:
    """Check that the arguments are valid."""
    lazy_import_playwright_browsers()
    if values.get('async_browser') is None and values.get('sync_browser') is None:
        raise ValueError('Either async_browser or sync_browser must be specified.')
    return values