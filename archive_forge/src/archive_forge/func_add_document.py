import redis
from ...asyncio.client import Pipeline as AsyncioPipeline
from .commands import (
def add_document(self, doc_id, nosave=False, score=1.0, payload=None, replace=False, partial=False, no_create=False, **fields):
    """
            Add a document to the batch query
            """
    self.client._add_document(doc_id, conn=self._pipeline, nosave=nosave, score=score, payload=payload, replace=replace, partial=partial, no_create=no_create, **fields)
    self.current_chunk += 1
    self.total += 1
    if self.current_chunk >= self.chunk_size:
        self.commit()