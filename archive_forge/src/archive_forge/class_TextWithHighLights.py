import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class TextWithHighLights(BaseModel, extra=Extra.allow):
    """Text with highlights."""
    Text: str
    'The text.'
    Highlights: Optional[Any]
    'The highlights.'