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
def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
    """Construct an SQL query for inserting data into the Clickhouse database.

        This method formats and constructs an SQL `INSERT` query string using the
        provided transaction data and column names. It is utilized internally during
        the process of batch insertion of documents and their embeddings into the
        database.

        Args:
            transac: iterable of tuples, representing a row of data to be inserted.
            column_names: iterable of strings representing the names of the columns
                into which data will be inserted.

        Returns:
            A string containing the constructed SQL `INSERT` query.
        """
    ks = ','.join(column_names)
    _data = []
    for n in transac:
        n = ','.join([f"'{self.escape_str(str(_n))}'" for _n in n])
        _data.append(f'({n})')
    i_str = f'\n                INSERT INTO TABLE \n                    {self.config.database}.{self.config.table}({ks})\n                VALUES\n                {','.join(_data)}\n                '
    return i_str