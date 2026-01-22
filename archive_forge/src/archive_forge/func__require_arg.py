from __future__ import annotations
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _require_arg(arg: Any, arg_name: str) -> None:
    """Raise ValueError if the required arg with name `arg_name` is None."""
    if not arg:
        raise ValueError(f'`{arg_name}` is required for this index.')