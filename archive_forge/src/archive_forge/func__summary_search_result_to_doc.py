from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
def _summary_search_result_to_doc(self, results: List[MemorySearchResult]) -> List[Document]:
    return [Document(page_content=r.summary.content, metadata={'score': r.dist, 'uuid': r.summary.uuid, 'created_at': r.summary.created_at, 'token_count': r.summary.token_count}) for r in results if r.summary]