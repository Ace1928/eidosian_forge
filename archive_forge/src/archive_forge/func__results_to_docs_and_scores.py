from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
    """Return docs and scores from results."""
    docs = [(Document(page_content=result[0], metadata=json.loads(result[1])), result[2] if self.embedding_function is not None else None) for result in results]
    return docs