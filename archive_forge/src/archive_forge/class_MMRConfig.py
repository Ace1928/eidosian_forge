from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
@dataclass
class MMRConfig:
    """Configuration for Maximal Marginal Relevance (MMR) search.

    is_enabled: True if MMR is enabled, False otherwise
    mmr_k: number of results to fetch for MMR, defaults to 50
    diversity_bias: number between 0 and 1 that determines the degree
        of diversity among the results with 0 corresponding
        to minimum diversity and 1 to maximum diversity.
        Defaults to 0.3.
        Note: diversity_bias is equivalent 1-lambda_mult
        where lambda_mult is the value often used in max_marginal_relevance_search()
        We chose to use that since we believe it's more intuitive to the user.
    """
    is_enabled: bool = False
    mmr_k: int = 50
    diversity_bias: float = 0.3