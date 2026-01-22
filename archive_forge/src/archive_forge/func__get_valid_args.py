from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def _get_valid_args(cls, method_name):
    if method_name == 'search':
        return cls._valid_search_kwargs
    else:
        return []