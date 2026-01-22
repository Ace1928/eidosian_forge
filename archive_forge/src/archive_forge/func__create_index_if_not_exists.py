import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _create_index_if_not_exists(self, dims_length: Optional[int]=None) -> None:
    """Create the index if it doesn't already exist.

        Args:
            dims_length: Length of the embedding vectors.
        """
    if self.client.indices.exists(index=self.index_name):
        logger.info(f'Index {self.index_name} already exists. Skipping creation.')
    else:
        if dims_length is None:
            raise ValueError('Cannot create index without specifying dims_length ' + "when the index doesn't already exist. ")
        indexMapping = self._index_mapping(dims_length=dims_length)
        logger.debug(f'Creating index {self.index_name} with mappings {indexMapping}')
        self.client.indices.create(index=self.index_name, body={'settings': {'index.knn': True, **self.index_settings}, 'mappings': {'properties': indexMapping}})