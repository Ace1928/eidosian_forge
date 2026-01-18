import logging
from typing import (
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
Return the Vector Store initialized from texts and embeddings.

        Args:
            cls (Type[VST]): The Vector Store class to use to initialize
                the Vector Store.
            texts (List[str]): The texts to initialize the Vector Store with.
            embedding (Embeddings): The embedding function to use.
            metadatas (Optional[List[dict]], optional): The metadata associated with
                the texts. Defaults to None.
            kwargs (Any): Vector Store specific parameters. The following are forwarded
                to the Vector Store constructor and required:
            - index_name (str, optional): The name of the index to store the documents
                in. Defaults to "default".
            - text_field (str, optional): The name of the metadata field to store the
                original text in. Defaults to "text".
            - distance_strategy (DistanceStrategy, optional): The distance strategy to
                use. Defaults to DistanceStrategy.COSINE. If you select
                DistanceStrategy.EUCLIDEAN_DISTANCE, Momento uses the squared
                Euclidean distance.
            - ensure_index_exists (bool, optional): Whether to ensure that the index
                exists before adding documents to it. Defaults to True.
            Additionally you can either pass in a client or an API key
            - client (PreviewVectorIndexClient): The Momento Vector Index client to use.
            - api_key (Optional[str]): The configuration to use to initialize
                the Vector Index with. Defaults to None. If None, the configuration
                is initialized from the environment variable `MOMENTO_API_KEY`.

        Returns:
            VST: Momento Vector Index vector store initialized from texts and
                embeddings.
        