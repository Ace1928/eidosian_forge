from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def __from(cls, config: KineticaSettings, texts: List[str], embeddings: List[List[float]], embedding: Embeddings, dimensions: int, metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, distance_strategy: DistanceStrategy=DEFAULT_DISTANCE_STRATEGY, pre_delete_collection: bool=False, logger: Optional[logging.Logger]=None, **kwargs: Any) -> Kinetica:
    """Class method to assist in constructing the `Kinetica` store instance
            using different combinations of parameters

        Args:
            config (KineticaSettings): a `KineticaSettings` instance
            texts (List[str]): The list of texts to generate embeddings for and store
            embeddings (List[List[float]]): List of embeddings
            embedding (Embeddings): the Embedding function
            dimensions (int): The number of dimensions the embeddings have
            metadatas (Optional[List[dict]], optional): List of JSON data associated
                        with each text. Defaults to None.
            ids (Optional[List[str]], optional): List of unique IDs (UUID by default)
                        associated with each text. Defaults to None.
            collection_name (str, optional): Kinetica schema name.
                        Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            distance_strategy (DistanceStrategy, optional): Not used for now.
                        Defaults to DEFAULT_DISTANCE_STRATEGY.
            pre_delete_collection (bool, optional): Whether to delete the Kinetica
                        schema or not. Defaults to False.
            logger (Optional[logging.Logger], optional): Logger to use for logging at
                        different levels. Defaults to None.

        Returns:
            Kinetica: An instance of Kinetica class
        """
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if not metadatas:
        metadatas = [{} for _ in texts]
    store = cls(config=config, collection_name=collection_name, embedding_function=embedding, distance_strategy=distance_strategy, pre_delete_collection=pre_delete_collection, logger=logger, **kwargs)
    store.__post_init__(dimensions)
    store.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs)
    return store