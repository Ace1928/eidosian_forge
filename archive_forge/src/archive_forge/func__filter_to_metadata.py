from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _filter_to_metadata(filter_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if filter_dict is None:
        return {}
    else:
        metadata_filter = {}
        for k, v in filter_dict.items():
            if k and k[0] == '$':
                if isinstance(v, list):
                    metadata_filter[k] = [AstraDB._filter_to_metadata(f) for f in v]
                else:
                    metadata_filter[k] = AstraDB._filter_to_metadata(v)
            else:
                metadata_filter[f'metadata.{k}'] = v
        return metadata_filter