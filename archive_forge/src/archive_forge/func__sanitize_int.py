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
def _sanitize_int(input_int: any) -> int:
    value = int(str(input_int))
    if value < -1:
        raise ValueError(f'Value ({value}) must not be smaller than -1')
    return int(str(input_int))