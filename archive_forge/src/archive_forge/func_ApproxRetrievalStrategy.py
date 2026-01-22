import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
@staticmethod
def ApproxRetrievalStrategy(query_model_id: Optional[str]=None, hybrid: Optional[bool]=False, rrf: Optional[Union[dict, bool]]=True) -> 'ApproxRetrievalStrategy':
    """Used to perform approximate nearest neighbor search
        using the HNSW algorithm.

        At build index time, this strategy will create a
        dense vector field in the index and store the
        embedding vectors in the index.

        At query time, the text will either be embedded using the
        provided embedding function or the query_model_id
        will be used to embed the text using the model
        deployed to Elasticsearch.

        if query_model_id is used, do not provide an embedding function.

        Args:
            query_model_id: Optional. ID of the model to use to
                            embed the query text within the stack. Requires
                            embedding model to be deployed to Elasticsearch.
            hybrid: Optional. If True, will perform a hybrid search
                    using both the knn query and a text query.
                    Defaults to False.
            rrf: Optional. rrf is Reciprocal Rank Fusion.
                 When `hybrid` is True,
                    and `rrf` is True, then rrf: {}.
                    and `rrf` is False, then rrf is omitted.
                    and isinstance(rrf, dict) is True, then pass in the dict values.
                 rrf could be passed for adjusting 'rank_constant' and 'window_size'.
        """
    return ApproxRetrievalStrategy(query_model_id=query_model_id, hybrid=hybrid, rrf=rrf)