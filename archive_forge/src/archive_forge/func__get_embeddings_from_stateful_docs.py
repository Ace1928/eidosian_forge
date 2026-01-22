from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
def _get_embeddings_from_stateful_docs(embeddings: Embeddings, documents: Sequence[_DocumentWithState]) -> List[List[float]]:
    if len(documents) and 'embedded_doc' in documents[0].state:
        embedded_documents = [doc.state['embedded_doc'] for doc in documents]
    else:
        embedded_documents = embeddings.embed_documents([d.page_content for d in documents])
        for doc, embedding in zip(documents, embedded_documents):
            doc.state['embedded_doc'] = embedding
    return embedded_documents