import logging
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
@staticmethod
def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
    return f'Page: {page_title}\nSummary: {wiki_page.summary}'