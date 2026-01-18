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
def write_schema(self, path: Union[str, os.PathLike]) -> None:
    """Write the schema to a yaml file."""
    with open(path, 'w+') as f:
        yaml.dump(self.schema, f)