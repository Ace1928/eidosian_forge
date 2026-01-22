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
def _handle_batch(document_batch: List[DocDict]) -> List[str]:
    im_result = self.collection.insert_many(documents=document_batch, options={'ordered': False}, partial_failures_allowed=True)
    batch_inserted, missing_from_batch = self._get_missing_from_batch(document_batch, im_result)

    def _handle_missing_document(missing_document: DocDict) -> str:
        replacement_result = self.collection.find_one_and_replace(filter={'_id': missing_document['_id']}, replacement=missing_document)
        return replacement_result['data']['document']['_id']
    _u_max_workers = overwrite_concurrency or self.bulk_insert_overwrite_concurrency
    with ThreadPoolExecutor(max_workers=_u_max_workers) as tpe2:
        batch_replaced = list(tpe2.map(_handle_missing_document, missing_from_batch))
    return batch_inserted + batch_replaced