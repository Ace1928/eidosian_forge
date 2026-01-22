from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
Returns a list of documents with their scores

        Args:
            embeddings: The query vector
            k: the number of documents to return
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw: available as a preview feature only,
                                   to enable visit https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/preview-features
            ef_search: The size of the dynamic candidate list for search
                       (40 by default). A higher value provides better
                       recall at the cost of speed.
            score_threshold: (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.
                Only vector-ivf search supports this for now.

        Returns:
            A list of documents closest to the query vector
        