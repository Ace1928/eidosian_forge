from __future__ import annotations
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _insert(self, texts: List[str], ids: Optional[List[str]], metadata: Optional[Any]=None) -> None:
    try:
        import numpy as np
    except ImportError:
        raise ImportError('Could not import numpy python package. Please install it with `pip install numpy`.')
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Could not import pandas python package. Please install it with `pip install pandas`.')
    embeds = self._embedding.embed_documents(texts)
    df = pd.DataFrame()
    df['id'] = ids
    df['text'] = [t.encode('utf-8') for t in texts]
    df['embeddings'] = [np.array(e, dtype='float32') for e in embeds]
    if metadata is not None:
        df = pd.concat([df, metadata], axis=1)
    self._table.insert(df, warn=False)