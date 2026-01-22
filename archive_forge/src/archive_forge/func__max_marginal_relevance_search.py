from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _max_marginal_relevance_search(self, embedding: list[float], k: int=4, fetch_k: int=20, lambda_mult: float=0.5, param: Optional[dict]=None, expr: Optional[str]=None, **kwargs: Any) -> List[Document]:
    """Perform a search and return results that are reordered by MMR."""
    ef = 10 if param is None else param.get('ef', 10)
    anns = self.mochowtable.AnnSearch(vector_field=self.field_vector, vector_floats=[float(num) for num in embedding], params=self.mochowtable.HNSWSearchParams(ef=ef, limit=k), filter=expr)
    res = self.table.search(anns=anns, retrieve_vector=True)
    documents: List[Document] = []
    ordered_result_embeddings = []
    rows = [[item] for item in res.rows]
    if rows is None or len(rows) == 0:
        return documents
    for row in rows:
        for result in row:
            row_data = result.get('row', {})
            meta = row_data.get(self.field_metadata)
            if meta is not None:
                meta = json.loads(meta)
            doc = Document(page_content=row_data.get(self.field_text), metadata=meta)
            documents.append(doc)
            ordered_result_embeddings.append(row_data.get(self.field_vector))
    new_ordering = maximal_marginal_relevance(np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult)
    ret = []
    for x in new_ordering:
        if x == -1:
            break
        else:
            ret.append(documents[x])
    return ret