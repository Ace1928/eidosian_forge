from typing import Any, List, Sequence
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.pydantic_v1 import BaseModel
def _litm_reordering(documents: List[Document]) -> List[Document]:
    """Lost in the middle reorder: the less relevant documents will be at the
    middle of the list and more relevant elements at beginning / end.
    See: https://arxiv.org/abs//2307.03172"""
    documents.reverse()
    reordered_result = []
    for i, value in enumerate(documents):
        if i % 2 == 1:
            reordered_result.append(value)
        else:
            reordered_result.insert(0, value)
    return reordered_result