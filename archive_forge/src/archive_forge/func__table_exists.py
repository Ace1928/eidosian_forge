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
def _table_exists(self, table_name) -> bool:
    sql_str = 'SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?'
    try:
        cur = self.connection.cursor()
        cur.execute(sql_str, table_name)
        if cur.has_result_set():
            rows = cur.fetchall()
            if rows[0][0] == 1:
                return True
    finally:
        cur.close()
    return False