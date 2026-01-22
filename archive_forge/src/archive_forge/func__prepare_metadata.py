from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for indexing in Redis by sanitizing its values.

    - String, integer, and float values remain unchanged.
    - None or empty values are replaced with empty strings.
    - Lists/tuples of strings are joined into a single string with a comma separator.

    Args:
        metadata (Dict[str, Any]): A dictionary where keys are metadata
            field names and values are the metadata values.

    Returns:
        Dict[str, Any]: A sanitized dictionary ready for indexing in Redis.

    Raises:
        ValueError: If any metadata value is not one of the known
            types (string, int, float, or list of strings).
    """

    def raise_error(key: str, value: Any) -> None:
        raise ValueError(f"Metadata value for key '{key}' must be a string, int, " + f'float, or list of strings. Got {type(value).__name__}')
    clean_meta: Dict[str, Union[str, float, int]] = {}
    for key, value in metadata.items():
        if value is None:
            clean_meta[key] = ''
            continue
        if isinstance(value, (str, int, float)):
            clean_meta[key] = value
        elif isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                clean_meta[key] = REDIS_TAG_SEPARATOR.join(value)
            else:
                raise_error(key, value)
        else:
            raise_error(key, value)
    return clean_meta