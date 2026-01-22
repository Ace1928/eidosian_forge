from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _query_result_to_docs(self, result: dict[str, Any]) -> List[Tuple[Document, float]]:
    documents = []
    for row in result['hits']:
        hit = row['hit'] or {}
        if self._output_fields is None:
            entity = hit['*']
        else:
            entity = {key: hit.get(key) for key in self._output_fields}
        doc = Document(page_content=self._to_content(entity), metadata=self._to_metadata(entity))
        documents.append((doc, hit['score()']))
    return documents