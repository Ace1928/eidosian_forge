from __future__ import annotations
import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _write_documents_to_rockset(self, batch: List[dict]) -> List[str]:
    add_doc_res = self._client.Documents.add_documents(collection=self._collection_name, data=batch, workspace=self._workspace)
    return [doc_status._id for doc_status in add_doc_res.data]