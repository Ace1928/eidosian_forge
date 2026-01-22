from __future__ import annotations
import uuid
import warnings
from itertools import repeat
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _add_vectors(client: supabase.client.Client, table_name: str, vectors: List[List[float]], documents: List[Document], ids: List[str], chunk_size: int) -> List[str]:
    """Add vectors to Supabase table."""
    rows: List[Dict[str, Any]] = [{'id': ids[idx], 'content': documents[idx].page_content, 'embedding': embedding, 'metadata': documents[idx].metadata} for idx, embedding in enumerate(vectors)]
    id_list: List[str] = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        result = client.from_(table_name).upsert(chunk).execute()
        if len(result.data) == 0:
            raise Exception('Error inserting: No rows added')
        ids = [str(i.get('id')) for i in result.data if i.get('id')]
        id_list.extend(ids)
    return id_list