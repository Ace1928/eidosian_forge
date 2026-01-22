from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
class _DocumentWithState(Document):
    """Wrapper for a document that includes arbitrary state."""
    state: dict = Field(default_factory=dict)
    'State associated with the document.'

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def to_document(self) -> Document:
        """Convert the DocumentWithState to a Document."""
        return Document(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_document(cls, doc: Document) -> '_DocumentWithState':
        """Create a DocumentWithState from a Document."""
        if isinstance(doc, cls):
            return doc
        return cls(page_content=doc.page_content, metadata=doc.metadata)