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
def _parse_float_array_from_string(array_as_string: str) -> List[float]:
    array_wo_brackets = array_as_string[1:-1]
    return [float(x) for x in array_wo_brackets.split(',')]