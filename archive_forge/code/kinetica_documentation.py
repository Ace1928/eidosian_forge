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
Adds the list of `Document` passed in to the vector store and returns it

        Args:
            cls (Type[Kinetica]): Kinetica class
            texts (List[str]): A list of texts for which the embeddings are generated
            embedding (Embeddings): List of embeddings
            config (KineticaSettings): a `KineticaSettings` instance
            metadatas (Optional[List[dict]], optional): List of dicts, JSON describing
                        the texts/documents. Defaults to None.
            collection_name (str, optional): Kinetica schema name.
                        Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            distance_strategy (DistanceStrategy, optional): Distance strategy
                        e.g., l2, cosine etc.. Defaults to DEFAULT_DISTANCE_STRATEGY.
            ids (Optional[List[str]], optional): A list of UUIDs for each text/document.
                        Defaults to None.
            pre_delete_collection (bool, optional): Indicates whether the Kinetica
                        schema is to be deleted or not. Defaults to False.

        Returns:
            Kinetica: a `Kinetica` instance
        