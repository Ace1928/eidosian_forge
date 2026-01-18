from __future__ import annotations
import time
from itertools import repeat
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def wait_for_indexing(self, timeout: float=5, ndocs: int=1) -> None:
    """Wait for the search index to contain a certain number of
        documents. Useful in tests.
        """
    start = time.time()
    while True:
        r = self._client.data().search_table(self._table_name, payload={'query': '', 'page': {'size': 0}})
        if r.status_code != 200:
            raise Exception(f'Error running search: {r.status_code} {r}')
        if r['totalCount'] == ndocs:
            break
        if time.time() - start > timeout:
            raise Exception('Timed out waiting for indexing to complete.')
        time.sleep(0.5)