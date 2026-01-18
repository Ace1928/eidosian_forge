import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
def weighted_reciprocal_rank(self, doc_lists: List[List[Document]]) -> List[Document]:
    """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
    if len(doc_lists) != len(self.weights):
        raise ValueError('Number of rank lists must be equal to the number of weights.')
    rrf_score: Dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(doc_lists, self.weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + self.c)
    all_docs = chain.from_iterable(doc_lists)
    sorted_docs = sorted(unique_by_key(all_docs, lambda doc: doc.page_content), reverse=True, key=lambda doc: rrf_score[doc.page_content])
    return sorted_docs