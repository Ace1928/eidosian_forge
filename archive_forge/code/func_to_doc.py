import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def to_doc(self, page_content_formatter: Callable[['ResultItem'], str]=combined_text) -> Document:
    """Converts this item to a Document."""
    page_content = page_content_formatter(self)
    metadata = self.get_additional_metadata()
    metadata.update({'result_id': self.Id, 'document_id': self.DocumentId, 'source': self.DocumentURI, 'title': self.get_title(), 'excerpt': self.get_excerpt(), 'document_attributes': self.get_document_attributes_dict(), 'score': self.get_score_attribute()})
    return Document(page_content=page_content, metadata=metadata)