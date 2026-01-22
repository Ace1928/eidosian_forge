from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore
@staticmethod
def _example_to_text(example: Dict[str, str], input_keys: Optional[List[str]]) -> str:
    if input_keys:
        return ' '.join(sorted_values({key: example[key] for key in input_keys}))
    else:
        return ' '.join(sorted_values(example))