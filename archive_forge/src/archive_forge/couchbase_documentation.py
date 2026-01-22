from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Construct a Couchbase vector store from a list of texts.

        Example:
            .. code-block:: python

            from langchain_community.vectorstores import CouchbaseVectorStore
            from langchain_openai import OpenAIEmbeddings

            from couchbase.cluster import Cluster
            from couchbase.auth import PasswordAuthenticator
            from couchbase.options import ClusterOptions
            from datetime import timedelta

            auth = PasswordAuthenticator(username, password)
            options = ClusterOptions(auth)
            connect_string = "couchbases://localhost"
            cluster = Cluster(connect_string, options)

            # Wait until the cluster is ready for use.
            cluster.wait_until_ready(timedelta(seconds=5))

            embeddings = OpenAIEmbeddings()

            texts = ["hello", "world"]

            vectorstore = CouchbaseVectorStore.from_texts(
                texts,
                embedding=embeddings,
                cluster=cluster,
                bucket_name="",
                scope_name="",
                collection_name="",
                index_name="vector-index",
            )

        Args:
            texts (List[str]): list of texts to add to the vector store.
            embedding (Embeddings): embedding function to use.
            metadatas (optional[List[Dict]): list of metadatas to add to documents.
            **kwargs: Keyword arguments used to initialize the vector store with and/or
                passed to `add_texts` method. Check the constructor and/or `add_texts`
                for the list of accepted arguments.

        Returns:
            A Couchbase vector store.

        