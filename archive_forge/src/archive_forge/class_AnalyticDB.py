from __future__ import annotations
import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from sqlalchemy import REAL, Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
class AnalyticDB(VectorStore):
    """`AnalyticDB` (distributed PostgreSQL) vector store.

    AnalyticDB is a distributed full postgresql syntax cloud-native database.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.

    """

    def __init__(self, connection_string: str, embedding_function: Embeddings, embedding_dimension: int=_LANGCHAIN_DEFAULT_EMBEDDING_DIM, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, pre_delete_collection: bool=False, logger: Optional[logging.Logger]=None, engine_args: Optional[dict]=None) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__(engine_args)

    def __post_init__(self, engine_args: Optional[dict]=None) -> None:
        """
        Initialize the store.
        """
        _engine_args = engine_args or {}
        if 'pool_recycle' not in _engine_args:
            _engine_args['pool_recycle'] = 3600
        self.engine = create_engine(self.connection_string, **_engine_args)
        self.create_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._euclidean_relevance_score_fn

    def create_table_if_not_exists(self) -> None:
        Table(self.collection_name, Base.metadata, Column('id', TEXT, primary_key=True, default=uuid.uuid4), Column('embedding', ARRAY(REAL)), Column('document', String, nullable=True), Column('metadata', JSON, nullable=True), extend_existing=True)
        with self.engine.connect() as conn:
            with conn.begin():
                Base.metadata.create_all(conn)
                index_name = f'{self.collection_name}_embedding_idx'
                index_query = text(f"\n                    SELECT 1\n                    FROM pg_indexes\n                    WHERE indexname = '{index_name}';\n                ")
                result = conn.execute(index_query).scalar()
                if not result:
                    index_statement = text(f'\n                        CREATE INDEX {index_name}\n                        ON {self.collection_name} USING ann(embedding)\n                        WITH (\n                            "dim" = {self.embedding_dimension},\n                            "hnsw_m" = 100\n                        );\n                    ')
                    conn.execute(index_statement)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.create_table_if_not_exists()

    def delete_collection(self) -> None:
        self.logger.debug('Trying to delete collection')
        drop_statement = text(f'DROP TABLE IF EXISTS {self.collection_name};')
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, batch_size: int=500, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding_function.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]
        chunks_table = Table(self.collection_name, Base.metadata, Column('id', TEXT, primary_key=True), Column('embedding', ARRAY(REAL)), Column('document', String, nullable=True), Column('metadata', JSON, nullable=True), extend_existing=True)
        chunks_table_data = []
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding in zip(texts, metadatas, ids, embeddings):
                    chunks_table_data.append({'id': chunk_id, 'embedding': embedding, 'document': document, 'metadata': metadata})
                    if len(chunks_table_data) == batch_size:
                        conn.execute(insert(chunks_table).values(chunks_table_data))
                        chunks_table_data.clear()
                if chunks_table_data:
                    conn.execute(insert(chunks_table).values(chunks_table_data))
        return ids

    def similarity_search(self, query: str, k: int=4, filter: Optional[dict]=None, **kwargs: Any) -> List[Document]:
        """Run similarity search with AnalyticDB with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(embedding=embedding, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int=4, filter: Optional[dict]=None) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(embedding=embedding, k=k, filter=filter)
        return docs

    def similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, filter: Optional[dict]=None) -> List[Tuple[Document, float]]:
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError("Could not import Row from sqlalchemy.engine. Please 'pip install sqlalchemy>=1.4'.")
        filter_condition = ''
        if filter is not None:
            conditions = [f'metadata->>{key!r} = {value!r}' for key, value in filter.items()]
            filter_condition = f'WHERE {' AND '.join(conditions)}'
        sql_query = f'\n            SELECT *, l2_distance(embedding, :embedding) as distance\n            FROM {self.collection_name}\n            {filter_condition}\n            ORDER BY embedding <-> :embedding\n            LIMIT :k\n        '
        params = {'embedding': embedding, 'k': k}
        with self.engine.connect() as conn:
            results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()
        documents_with_scores = [(Document(page_content=result.document, metadata=result.metadata), result.distance if self.embedding_function is not None else None) for result in results]
        return documents_with_scores

    def similarity_search_by_vector(self, embedding: List[float], k: int=4, filter: Optional[dict]=None, **kwargs: Any) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(embedding=embedding, k=k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def delete(self, ids: Optional[List[str]]=None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        if ids is None:
            raise ValueError('No ids provided to delete.')
        chunks_table = Table(self.collection_name, Base.metadata, Column('id', TEXT, primary_key=True), Column('embedding', ARRAY(REAL)), Column('document', String, nullable=True), Column('metadata', JSON, nullable=True), extend_existing=True)
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = chunks_table.c.id.in_(ids)
                    conn.execute(chunks_table.delete().where(delete_condition))
                    return True
        except Exception as e:
            print('Delete operation failed:', str(e))
            return False

    @classmethod
    def from_texts(cls: Type[AnalyticDB], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, embedding_dimension: int=_LANGCHAIN_DEFAULT_EMBEDDING_DIM, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, ids: Optional[List[str]]=None, pre_delete_collection: bool=False, engine_args: Optional[dict]=None, **kwargs: Any) -> AnalyticDB:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """
        connection_string = cls.get_connection_string(kwargs)
        store = cls(connection_string=connection_string, collection_name=collection_name, embedding_function=embedding, embedding_dimension=embedding_dimension, pre_delete_collection=pre_delete_collection, engine_args=engine_args)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(data=kwargs, key='connection_string', env_key='PG_CONNECTION_STRING')
        if not connection_string:
            raise ValueError('Postgres connection string is requiredEither pass it as a parameteror set the PG_CONNECTION_STRING environment variable.')
        return connection_string

    @classmethod
    def from_documents(cls: Type[AnalyticDB], documents: List[Document], embedding: Embeddings, embedding_dimension: int=_LANGCHAIN_DEFAULT_EMBEDDING_DIM, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, ids: Optional[List[str]]=None, pre_delete_collection: bool=False, engine_args: Optional[dict]=None, **kwargs: Any) -> AnalyticDB:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)
        kwargs['connection_string'] = connection_string
        return cls.from_texts(texts=texts, pre_delete_collection=pre_delete_collection, embedding=embedding, embedding_dimension=embedding_dimension, metadatas=metadatas, ids=ids, collection_name=collection_name, engine_args=engine_args, **kwargs)

    @classmethod
    def connection_string_from_db_params(cls, driver: str, host: str, port: int, database: str, user: str, password: str) -> str:
        """Return connection string from database parameters."""
        return f'postgresql+{driver}://{user}:{password}@{host}:{port}/{database}'