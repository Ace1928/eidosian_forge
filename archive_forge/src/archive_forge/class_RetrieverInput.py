from functools import partial
from typing import Optional
from langchain_core.callbacks.manager import (
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain.tools import Tool
class RetrieverInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description='query to look up in retriever')