from typing import Any, Callable, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class ColumnNotFoundError(Exception):
    """Column not found error."""

    def __init__(self, missing_key: str, query: str):
        super().__init__(f'Column "{missing_key}" not selected in query:\n{query}')