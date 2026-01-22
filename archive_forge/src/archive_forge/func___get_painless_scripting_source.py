from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __get_painless_scripting_source(space_type: str, vector_field: str='vector_field') -> str:
    """For Painless Scripting, it returns the script source based on space type."""
    source_value = '(1.0 + ' + space_type + "(params.query_value, doc['" + vector_field + "']))"
    if space_type == 'cosineSimilarity':
        return source_value
    else:
        return '1/' + source_value