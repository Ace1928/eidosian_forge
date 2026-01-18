from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
Construct Meilisearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Meilisearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Meilisearch
                from langchain_community.embeddings import OpenAIEmbeddings
                import meilisearch

                # The environment should be the one specified next to the API key
                # in your Meilisearch console
                client = meilisearch.Client(url='http://127.0.0.1:7700', api_key='***')
                embedding = OpenAIEmbeddings()
                embedders: Embedders index setting.
                embedder_name: Name of the embedder. Defaults to "default".
                docsearch = Meilisearch.from_texts(
                    client=client,
                    embedding=embedding,
                )
        