from __future__ import annotations
import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
@staticmethod
def _euclidean_relevance_score_fn(distance: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - distance / math.sqrt(2)