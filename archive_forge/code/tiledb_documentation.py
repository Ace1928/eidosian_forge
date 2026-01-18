from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
Load a TileDB index from a URI.

        Args:
            index_uri: The URI of the TileDB vector index.
            embedding: Embeddings to use when generating queries.
            metric: Optional, Metric to use for indexing. Defaults to "euclidean".
            config: Optional, TileDB config
            timestamp: Optional, timestamp to use for opening the arrays.
        