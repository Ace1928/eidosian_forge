import importlib.util
import os
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
Embed a list of text documents using the Optimized Embedder model.

        Input:
            texts: List[str] = List of text documents to embed.
        Output:
            List[List[float]] = The embeddings of each text document.
        