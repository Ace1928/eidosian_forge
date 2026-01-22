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
def as_retriever(self, **kwargs: Any) -> RedisVectorStoreRetriever:
    tags = kwargs.pop('tags', None) or []
    tags.extend(self._get_retriever_tags())
    return RedisVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)