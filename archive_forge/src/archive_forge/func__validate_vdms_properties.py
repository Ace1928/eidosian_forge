from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _validate_vdms_properties(metadata: Dict[str, Any]) -> Dict:
    new_metadata: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(value, list):
            new_metadata[str(key)] = value
    return new_metadata