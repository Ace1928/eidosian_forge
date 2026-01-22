from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _select_relevance_score_fn(self) -> Callable[[float], float]:
    if self.relevance_score_fn:
        return self.relevance_score_fn
    metric_map = {'COSINE': self._cosine_relevance_score_fn, 'IP': self._max_inner_product_relevance_score_fn, 'L2': self._euclidean_relevance_score_fn}
    try:
        return metric_map[self._schema.content_vector.distance_metric]
    except KeyError:
        return _default_relevance_score