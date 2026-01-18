from __future__ import annotations
import concurrent.futures
from typing import Any, Iterable, List, Optional
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
Configuration for this pydantic object.