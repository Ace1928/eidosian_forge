from enum import Enum
from typing import Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from langchain.storage._lc_store import create_kv_docstore
Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        