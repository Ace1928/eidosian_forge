from __future__ import annotations
import importlib.util
import json
import re
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
def _check_column(self, table_name, column_name, column_type, column_length=None):
    sql_str = 'SELECT DATA_TYPE_NAME, LENGTH FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ? AND COLUMN_NAME = ?'
    try:
        cur = self.connection.cursor()
        cur.execute(sql_str, (table_name, column_name))
        if cur.has_result_set():
            rows = cur.fetchall()
            if len(rows) == 0:
                raise AttributeError(f'Column {column_name} does not exist')
            if rows[0][0] not in column_type:
                raise AttributeError(f'Column {column_name} has the wrong type: {rows[0][0]}')
            if column_length is not None:
                if rows[0][1] != column_length:
                    raise AttributeError(f'Column {column_name} has the wrong length: {rows[0][1]}')
        else:
            raise AttributeError(f'Column {column_name} does not exist')
    finally:
        cur.close()