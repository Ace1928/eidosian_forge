import logging
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.vertexai import _VertexAICommon
from langchain_community.utilities.vertexai import raise_vertex_import_error
@staticmethod
def _split_by_punctuation(text: str) -> List[str]:
    """Splits a string by punctuation and whitespace characters."""
    split_by = string.punctuation + '\t\n '
    pattern = f'([{split_by}])'
    return [segment for segment in re.split(pattern, text) if segment]