from __future__ import annotations
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _extract_fields(self) -> None:
    """Grab the existing fields from the Collection"""
    from pymilvus import Collection
    if isinstance(self.col, Collection):
        schema = self.col.schema
        for x in schema.fields:
            self.fields.append(x.name)