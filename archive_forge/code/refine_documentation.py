from __future__ import annotations
from typing import Any, Dict, List, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain.chains.combine_documents.base import (
from langchain.chains.llm import LLMChain
Async combine by mapping a first chain over all, then stuffing
         into a final chain.

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        