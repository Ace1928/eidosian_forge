from __future__ import annotations
from typing import Any, Dict, List, Tuple, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
class HeaderType(TypedDict):
    """Header type as typed dict."""
    level: int
    name: str
    data: str