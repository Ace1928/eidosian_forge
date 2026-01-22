import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _index_mapping(self, dims_length: Union[int, None]) -> Dict:
    """
        Executes when the index is created.

        Args:
            dims_length: Numeric length of the embedding vectors,
                        or None if not using vector-based query.
            index_params: The extra pamameters for creating index.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """
    model = self.vector_params.get('model', '')
    if 'lsh' == model:
        mapping: Dict[Any, Any] = {self.vector_field: {'type': self.vector_type, 'knn': {'dims': dims_length, 'model': 'lsh', 'similarity': self.vector_params.get('similarity', 'cosine'), 'L': self.vector_params.get('L', 99), 'k': self.vector_params.get('k', 1)}}}
        if mapping[self.vector_field]['knn']['similarity'] == 'l2':
            mapping[self.vector_field]['knn']['w'] = self.vector_params.get('w', 3)
        return mapping
    elif 'permutation_lsh' == model:
        return {self.vector_field: {'type': self.vector_type, 'knn': {'dims': dims_length, 'model': 'permutation_lsh', 'k': self.vector_params.get('k', 10), 'similarity': self.vector_params.get('similarity', 'cosine'), 'repeating': self.vector_params.get('repeating', True)}}}
    else:
        return {self.vector_field: {'type': self.vector_type, 'knn': {'dims': dims_length}}}