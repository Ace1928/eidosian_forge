import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
@staticmethod
def SparseVectorRetrievalStrategy(model_id: Optional[str]=None) -> 'SparseRetrievalStrategy':
    """Used to perform sparse vector search via text_expansion.
        Used for when you want to use ELSER model to perform document search.

        At build index time, this strategy will create a pipeline that
        will embed the text using the ELSER model and store the
        resulting tokens in the index.

        At query time, the text will be embedded using the ELSER
        model and the resulting tokens will be used to
        perform a text_expansion query.

        Args:
            model_id: Optional. Default is ".elser_model_1".
                    ID of the model to use to embed the query text
                    within the stack. Requires embedding model to be
                    deployed to Elasticsearch.
        """
    return SparseRetrievalStrategy(model_id=model_id)