import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class DocumentAttribute(BaseModel, extra=Extra.allow):
    """Document attribute."""
    Key: str
    'The key of the attribute.'
    Value: DocumentAttributeValue
    'The value of the attribute.'