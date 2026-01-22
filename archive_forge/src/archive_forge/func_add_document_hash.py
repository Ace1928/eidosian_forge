import redis
from ...asyncio.client import Pipeline as AsyncioPipeline
from .commands import (
def add_document_hash(self, doc_id, score=1.0, replace=False):
    """
            Add a hash to the batch query
            """
    self.client._add_document_hash(doc_id, conn=self._pipeline, score=score, replace=replace)
    self.current_chunk += 1
    self.total += 1
    if self.current_chunk >= self.chunk_size:
        self.commit()