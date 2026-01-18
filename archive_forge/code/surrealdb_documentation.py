import asyncio
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Create SurrealDBStore from list of text

        Args:
            texts (List[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
            dburl (str): SurrealDB connection url
            ns (str): surrealdb namespace for the vector store.
                (default: "langchain")
            db (str): surrealdb database for the vector store.
                (default: "database")
            collection (str): surrealdb collection for the vector store.
                (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use.