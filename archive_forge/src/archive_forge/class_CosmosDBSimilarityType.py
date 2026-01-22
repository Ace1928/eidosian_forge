from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class CosmosDBSimilarityType(str, Enum):
    """Cosmos DB Similarity Type as enumerator."""
    COS = 'COS'
    'CosineSimilarity'
    IP = 'IP'
    'inner - product'
    L2 = 'L2'
    'Euclidean distance'