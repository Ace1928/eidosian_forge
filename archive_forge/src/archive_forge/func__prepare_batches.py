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
def _prepare_batches(texts: List[str], batch_size: int) -> List[List[str]]:
    """Splits texts in batches based on current maximum batch size
        and maximum tokens per request.
        """
    text_index = 0
    texts_len = len(texts)
    batch_token_len = 0
    batches: List[List[str]] = []
    current_batch: List[str] = []
    if texts_len == 0:
        return []
    while text_index < texts_len:
        current_text = texts[text_index]
        current_text_token_cnt = len(VertexAIEmbeddings._split_by_punctuation(current_text)) * 2
        end_of_batch = False
        if current_text_token_cnt > _MAX_TOKENS_PER_BATCH:
            if len(current_batch) > 0:
                batches.append(current_batch)
            current_batch = [current_text]
            text_index += 1
            end_of_batch = True
        elif batch_token_len + current_text_token_cnt > _MAX_TOKENS_PER_BATCH or len(current_batch) == batch_size:
            end_of_batch = True
        else:
            if text_index == texts_len - 1:
                end_of_batch = True
            batch_token_len += current_text_token_cnt
            current_batch.append(current_text)
            text_index += 1
        if end_of_batch:
            batches.append(current_batch)
            current_batch = []
            batch_token_len = 0
    return batches