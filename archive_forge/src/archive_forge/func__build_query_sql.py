from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
def _build_query_sql(self, q_emb: List[float], topk: int, where_str: Optional[str]=None) -> str:
    """Construct an SQL query for performing a similarity search.

        This internal method generates an SQL query for finding the top-k most similar
        vectors in the database to a given query vector.It allows for optional filtering
        conditions to be applied via a WHERE clause.

        Args:
            q_emb: The query vector as a list of floats.
            topk: The number of top similar items to retrieve.
            where_str: opt str representing additional WHERE conditions for the query
                Defaults to None.

        Returns:
            A string containing the SQL query for the similarity search.
        """
    q_emb_str = ','.join(map(str, q_emb))
    if where_str:
        where_str = f'PREWHERE {where_str}'
    else:
        where_str = ''
    settings_strs = []
    if self.config.index_query_params:
        for k in self.config.index_query_params:
            settings_strs.append(f'SETTING {k}={self.config.index_query_params[k]}')
    q_str = f'\n            SELECT {self.config.column_map['document']}, \n                {self.config.column_map['metadata']}, dist\n            FROM {self.config.database}.{self.config.table}\n            {where_str}\n            ORDER BY L2Distance({self.config.column_map['embedding']}, [{q_emb_str}]) \n                AS dist {self.dist_order}\n            LIMIT {topk} {' '.join(settings_strs)}\n            '
    return q_str