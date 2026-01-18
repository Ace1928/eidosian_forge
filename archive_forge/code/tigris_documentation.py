from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Return VectorStore initialized from texts and embeddings.