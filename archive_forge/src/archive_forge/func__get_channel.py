from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_channel(self) -> Any:
    try:
        import grpc
    except ImportError:
        raise ValueError('Could not import grpcio python package. Please install it with `pip install grpcio`.')
    return grpc.secure_channel(self.target, self.grpc_credentials, options=self.grpc_options) if self.grpc_use_secure else grpc.insecure_channel(self.target, options=self.grpc_options)