from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
    """Filter that drops redundant documents by comparing their embeddings."""
    embeddings: Embeddings
    'Embeddings to use for embedding document contents.'
    similarity_fn: Callable = cosine_similarity
    'Similarity function for comparing documents. Function expected to take as input\n    two matrices (List[List[float]]) and return a matrix of scores where higher values\n    indicate greater similarity.'
    similarity_threshold: float = 0.95
    'Threshold for determining when two documents are similar enough\n    to be considered redundant.'

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Filter down documents."""
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(self.embeddings, stateful_documents)
        included_idxs = _filter_similar_embeddings(embedded_documents, self.similarity_fn, self.similarity_threshold)
        return [stateful_documents[i] for i in sorted(included_idxs)]