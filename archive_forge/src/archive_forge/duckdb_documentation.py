from __future__ import annotations
import json
import uuid
from typing import Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
Ensures the table for storing embeddings exists.