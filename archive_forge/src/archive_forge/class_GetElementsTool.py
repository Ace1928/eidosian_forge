from __future__ import annotations
import json
from typing import TYPE_CHECKING, List, Optional, Sequence, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
class GetElementsTool(BaseBrowserTool):
    """Tool for getting elements in the current web page matching a CSS selector."""
    name: str = 'get_elements'
    description: str = 'Retrieve elements in the current web page matching the given CSS selector'
    args_schema: Type[BaseModel] = GetElementsToolInput

    def _run(self, selector: str, attributes: Sequence[str]=['innerText'], run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f'Synchronous browser not provided to {self.name}')
        page = get_current_page(self.sync_browser)
        results = _get_elements(page, selector, attributes)
        return json.dumps(results, ensure_ascii=False)

    async def _arun(self, selector: str, attributes: Sequence[str]=['innerText'], run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f'Asynchronous browser not provided to {self.name}')
        page = await aget_current_page(self.async_browser)
        results = await _aget_elements(page, selector, attributes)
        return json.dumps(results, ensure_ascii=False)