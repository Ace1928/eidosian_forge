from typing import Any, List, Sequence
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.pydantic_v1 import BaseModel
class LongContextReorder(BaseDocumentTransformer, BaseModel):
    """Reorder long context.

    Lost in the middle:
    Performance degrades when models must access relevant information
    in the middle of long contexts.
    See: https://arxiv.org/abs//2307.03172"""

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Reorders documents."""
        return _litm_reordering(list(documents))

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        raise NotImplementedError