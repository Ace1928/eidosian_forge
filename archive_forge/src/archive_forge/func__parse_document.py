from __future__ import annotations
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _parse_document(self, data: dict) -> Document:
    return Document(page_content=data.pop(self._text_field), metadata=data.pop(self._metadata_field) if self._metadata_field else data)