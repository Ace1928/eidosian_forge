from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _get_unsupported_items(kwargs, valid_items):
    kwargs = {k: v for k, v in kwargs.items() if k not in valid_items}
    unsupported_items = None
    if kwargs:
        unsupported_items = '`, `'.join(set(kwargs.keys()))
    return unsupported_items