from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _search_tql(self, tql: Optional[str], exec_option: Optional[str]=None, **kwargs: Any) -> List[Document]:
    """Function for performing tql_search.

        Args:
            tql (str): TQL Query string for direct evaluation.
                Available only for `compute_engine` and `tensor_db`.
            exec_option (str, optional): Supports 3 ways to search.
                Could be "python", "compute_engine" or "tensor_db". Default is "python".
                - ``python`` - Pure-python implementation for the client.
                    WARNING: not recommended for big datasets due to potential memory
                    issues.
                - ``compute_engine`` - C++ implementation of Deep Lake Compute
                    Engine for the client. Not for in-memory or local datasets.
                - ``tensor_db`` - Hosted Managed Tensor Database for storage
                    and query execution. Only for data in Deep Lake Managed Database.
                        Use runtime = {"db_engine": True} during dataset creation.
            return_score (bool): Return score with document. Default is False.

        Returns:
            Tuple[List[Document], List[Tuple[Document, float]]] - A tuple of two lists.
                The first list contains Documents, and the second list contains
                tuples of Document and float score.

        Raises:
            ValueError: If return_score is True but some condition is not met.
        """
    result = self.vectorstore.search(query=tql, exec_option=exec_option)
    metadatas = result['metadata']
    texts = result['text']
    docs = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    if kwargs:
        unsupported_argument = next(iter(kwargs))
        if kwargs[unsupported_argument] is not False:
            raise ValueError(f'specifying {unsupported_argument} is not supported with tql search.')
    return docs