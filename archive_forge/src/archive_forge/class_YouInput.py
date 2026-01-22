from typing import List, Optional, Type
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.you import YouSearchAPIWrapper
class YouInput(BaseModel):
    """Input schema for the you.com tool."""
    query: str = Field(description='should be a search query')