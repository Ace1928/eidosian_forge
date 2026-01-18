import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.math import cosine_similarity
from langchain_community.vectorstores.utils import maximal_marginal_relevance
In-memory implementation of VectorStore using a dictionary.
    Uses numpy to compute cosine similarity for search.

    Args:
        embedding:  embedding function to use.
    