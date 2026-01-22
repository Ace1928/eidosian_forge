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
def _sanitize_metadata_keys(metadata: dict) -> dict:
    for key in metadata.keys():
        if not HanaDB._compiled_pattern.match(key):
            raise ValueError(f'Invalid metadata key {key}')
    return metadata