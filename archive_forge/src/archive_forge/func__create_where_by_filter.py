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
def _create_where_by_filter(self, filter):
    query_tuple = []
    where_str = ''
    if filter:
        for i, key in enumerate(filter.keys()):
            if i == 0:
                where_str += ' WHERE '
            else:
                where_str += ' AND '
            where_str += f" JSON_VALUE({self.metadata_column}, '$.{key}') = ?"
            if isinstance(filter[key], bool):
                if filter[key]:
                    query_tuple.append('true')
                else:
                    query_tuple.append('false')
            elif isinstance(filter[key], int) or isinstance(filter[key], str):
                query_tuple.append(filter[key])
            else:
                raise ValueError(f'Unsupported filter data-type: {type(filter[key])}')
    return (where_str, query_tuple)