import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _query_body(self, query_vector: Union[List[float], None], filter: Optional[dict]=None, search_params: Dict={}) -> Dict:
    query_vector_body = {'field': search_params.get('vector_field', self.vector_field)}
    if self.vector_type == 'knn_dense_float_vector':
        query_vector_body['vec'] = {'values': query_vector}
        specific_params = self.get_dense_specific_model_similarity_params(search_params)
        query_vector_body.update(specific_params)
    else:
        query_vector_body['vec'] = {'true_indices': query_vector, 'total_indices': len(query_vector) if query_vector is not None else 0}
        specific_params = self.get_sparse_specific_model_similarity_params(search_params)
        query_vector_body.update(specific_params)
    query_vector_body = {'knn_nearest_neighbors': query_vector_body}
    if filter is not None and len(filter) != 0:
        query_vector_body = {'function_score': {'query': filter, 'functions': [query_vector_body]}}
    return {'size': search_params.get('size', 4), 'query': query_vector_body}