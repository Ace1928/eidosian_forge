import uuid
from itertools import islice
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.qdrant import Qdrant, QdrantException
def _generate_rest_batches(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[Sequence[str]]=None, batch_size: int=64) -> Generator[Tuple[List[str], List[Any]], None, None]:
    from qdrant_client import models as rest
    texts_iterator = iter(texts)
    metadatas_iterator = iter(metadatas or [])
    ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
    while (batch_texts := list(islice(texts_iterator, batch_size))):
        batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
        batch_ids = list(islice(ids_iterator, batch_size))
        batch_embeddings: List[Tuple[List[int], List[float]]] = [self.sparse_encoder(text) for text in batch_texts]
        points = [rest.PointStruct(id=point_id, vector={self.sparse_vector_name: rest.SparseVector(indices=sparse_vector[0], values=sparse_vector[1])}, payload=payload) for point_id, sparse_vector, payload in zip(batch_ids, batch_embeddings, Qdrant._build_payloads(batch_texts, batch_metadatas, self.content_payload_key, self.metadata_payload_key))]
        yield (batch_ids, points)