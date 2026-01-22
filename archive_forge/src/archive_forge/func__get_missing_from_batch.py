from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _get_missing_from_batch(document_batch: List[DocDict], insert_result: Dict[str, Any]) -> Tuple[List[str], List[DocDict]]:
    if 'status' not in insert_result:
        raise ValueError(f'API Exception while running bulk insertion: {str(insert_result)}')
    batch_inserted = insert_result['status']['insertedIds']
    missed_inserted_ids = {document['_id'] for document in document_batch} - set(batch_inserted)
    errors = insert_result.get('errors', [])
    num_errors = len(errors)
    unexpected_errors = any((error.get('errorCode') != 'DOCUMENT_ALREADY_EXISTS' for error in errors))
    if num_errors != len(missed_inserted_ids) or unexpected_errors:
        raise ValueError(f'API Exception while running bulk insertion: {str(errors)}')
    missing_from_batch = [document for document in document_batch if document['_id'] in missed_inserted_ids]
    return (batch_inserted, missing_from_batch)