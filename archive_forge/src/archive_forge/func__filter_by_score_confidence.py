import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def _filter_by_score_confidence(self, docs: List[Document]) -> List[Document]:
    """
        Filter out the records that have a score confidence
        greater than the required threshold.
        """
    if not self.min_score_confidence:
        return docs
    filtered_docs = [item for item in docs if item.metadata.get('score') is not None and isinstance(item.metadata['score'], str) and (KENDRA_CONFIDENCE_MAPPING.get(item.metadata['score'], 0.0) >= self.min_score_confidence)]
    return filtered_docs