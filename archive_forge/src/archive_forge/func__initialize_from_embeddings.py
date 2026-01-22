from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import sqlalchemy
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@classmethod
def _initialize_from_embeddings(cls, texts: List[str], embeddings: List[List[float]], embedding: Embeddings, metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, pre_delete_collection: bool=False, **kwargs: Any) -> PGEmbedding:
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if not metadatas:
        metadatas = [{} for _ in texts]
    connection_string = cls.get_connection_string(kwargs)
    store = cls(connection_string=connection_string, collection_name=collection_name, embedding_function=embedding, pre_delete_collection=pre_delete_collection)
    store.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs)
    return store