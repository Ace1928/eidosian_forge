from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Type
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.llm import LLMChain
Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        