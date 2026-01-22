from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
    """Embed search texts.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            List of floats representing the texts embedding.
        """
    if self.embeddings is not None:
        embeddings = self.embeddings.embed_documents(list(texts))
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
    elif self._embeddings_function is not None:
        embeddings = []
        for text in texts:
            embedding = self._embeddings_function(text)
            if hasattr(embeddings, 'tolist'):
                embedding = embedding.tolist()
            embeddings.append(embedding)
    else:
        raise ValueError('Neither of embeddings or embedding_function is set')
    return embeddings