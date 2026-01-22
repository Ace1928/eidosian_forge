from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore
def _documents_to_examples(self, documents: List[Document]) -> List[dict]:
    examples = [dict(e.metadata) for e in documents]
    if self.example_keys:
        examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
    return examples