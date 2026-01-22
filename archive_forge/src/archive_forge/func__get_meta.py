from __future__ import annotations
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_meta(self, result: Dict) -> Dict:
    """Get metadata from the result."""
    if self.meta_fields:
        return {field.name: result.get(field.name) for field in self.meta_fields}
    elif result.get(self.field_metadata):
        raw_meta = result.get(self.field_metadata)
        if raw_meta and isinstance(raw_meta, str):
            return json.loads(raw_meta)
    return {}