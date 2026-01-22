from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
class AzureSearch(VectorStore):
    """`Azure Cognitive Search` vector store."""

    def __init__(self, azure_search_endpoint: str, azure_search_key: str, index_name: str, embedding_function: Union[Callable, Embeddings], search_type: str='hybrid', semantic_configuration_name: Optional[str]=None, fields: Optional[List[SearchField]]=None, vector_search: Optional[VectorSearch]=None, semantic_configurations: Optional[Union[SemanticConfiguration, List[SemanticConfiguration]]]=None, scoring_profiles: Optional[List[ScoringProfile]]=None, default_scoring_profile: Optional[str]=None, cors_options: Optional[CorsOptions]=None, **kwargs: Any):
        from azure.search.documents.indexes.models import SearchableField, SearchField, SearchFieldDataType, SimpleField
        'Initialize with necessary components.'
        self.embedding_function = embedding_function
        if isinstance(self.embedding_function, Embeddings):
            self.embed_query = self.embedding_function.embed_query
        else:
            self.embed_query = self.embedding_function
        default_fields = [SimpleField(name=FIELDS_ID, type=SearchFieldDataType.String, key=True, filterable=True), SearchableField(name=FIELDS_CONTENT, type=SearchFieldDataType.String), SearchField(name=FIELDS_CONTENT_VECTOR, type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=len(self.embed_query('Text')), vector_search_profile_name='myHnswProfile'), SearchableField(name=FIELDS_METADATA, type=SearchFieldDataType.String)]
        user_agent = 'langchain'
        if 'user_agent' in kwargs and kwargs['user_agent']:
            user_agent += ' ' + kwargs['user_agent']
        self.client = _get_search_client(azure_search_endpoint, azure_search_key, index_name, semantic_configuration_name=semantic_configuration_name, fields=fields, vector_search=vector_search, semantic_configurations=semantic_configurations, scoring_profiles=scoring_profiles, default_scoring_profile=default_scoring_profile, default_fields=default_fields, user_agent=user_agent, cors_options=cors_options)
        self.search_type = search_type
        self.semantic_configuration_name = semantic_configuration_name
        self.fields = fields if fields else default_fields

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, **kwargs: Any) -> List[str]:
        """Add texts data to an existing index."""
        keys = kwargs.get('keys')
        ids = []
        if isinstance(self.embedding_function, Embeddings):
            try:
                embeddings = self.embedding_function.embed_documents(texts)
            except NotImplementedError:
                embeddings = [self.embedding_function.embed_query(x) for x in texts]
        else:
            embeddings = [self.embedding_function(x) for x in texts]
        if len(embeddings) == 0:
            logger.debug('Nothing to insert, skipping.')
            return []
        data = []
        for i, text in enumerate(texts):
            key = keys[i] if keys else str(uuid.uuid4())
            key = base64.urlsafe_b64encode(bytes(key, 'utf-8')).decode('ascii')
            metadata = metadatas[i] if metadatas else {}
            doc = {'@search.action': 'upload', FIELDS_ID: key, FIELDS_CONTENT: text, FIELDS_CONTENT_VECTOR: np.array(embeddings[i], dtype=np.float32).tolist(), FIELDS_METADATA: json.dumps(metadata)}
            if metadata:
                additional_fields = {k: v for k, v in metadata.items() if k in [x.name for x in self.fields]}
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=data)
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                data = []
        if len(data) == 0:
            return ids
        response = self.client.upload_documents(documents=data)
        if all([r.succeeded for r in response]):
            return ids
        else:
            raise Exception(response)

    def similarity_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        search_type = kwargs.get('search_type', self.search_type)
        if search_type == 'similarity':
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == 'hybrid':
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == 'semantic_hybrid':
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f'search_type of {search_type} not allowed.')
        return docs

    def similarity_search_with_relevance_scores(self, query: str, k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
        score_threshold = kwargs.pop('score_threshold', None)
        result = self.vector_search_with_score(query, k=k, **kwargs)
        return result if score_threshold is None else [r for r in result if r[1] >= score_threshold]

    def vector_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.vector_search_with_score(query, k=k, filters=kwargs.get('filters', None))
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(self, query: str, k: int=4, filters: Optional[str]=None) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery
        results = self.client.search(search_text='', vector_queries=[VectorizedQuery(vector=np.array(self.embed_query(query), dtype=np.float32).tolist(), k_nearest_neighbors=k, fields=FIELDS_CONTENT_VECTOR)], filter=filters, top=k)
        docs = [(Document(page_content=result.pop(FIELDS_CONTENT), metadata=json.loads(result[FIELDS_METADATA]) if FIELDS_METADATA in result else {k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR}), float(result['@search.score'])) for result in results]
        return docs

    def hybrid_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.hybrid_search_with_score(query, k=k, filters=kwargs.get('filters', None))
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(self, query: str, k: int=4, filters: Optional[str]=None) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery
        results = self.client.search(search_text=query, vector_queries=[VectorizedQuery(vector=np.array(self.embed_query(query), dtype=np.float32).tolist(), k_nearest_neighbors=k, fields=FIELDS_CONTENT_VECTOR)], filter=filters, top=k)
        docs = [(Document(page_content=result.pop(FIELDS_CONTENT), metadata=json.loads(result[FIELDS_METADATA]) if FIELDS_METADATA in result else {k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR}), float(result['@search.score'])) for result in results]
        return docs

    def semantic_hybrid_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(query, k=k, filters=kwargs.get('filters', None))
        return [doc for doc, _, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(self, query: str, k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(query, k=k, filters=kwargs.get('filters', None))
        return [(doc, score) for doc, score, _ in docs_and_scores]

    def semantic_hybrid_search_with_score_and_rerank(self, query: str, k: int=4, filters: Optional[str]=None) -> List[Tuple[Document, float, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery
        results = self.client.search(search_text=query, vector_queries=[VectorizedQuery(vector=np.array(self.embed_query(query), dtype=np.float32).tolist(), k_nearest_neighbors=k, fields=FIELDS_CONTENT_VECTOR)], filter=filters, query_type='semantic', semantic_configuration_name=self.semantic_configuration_name, query_caption='extractive', query_answer='extractive', top=k)
        semantic_answers = results.get_answers() or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {'text': semantic_answer.text, 'highlights': semantic_answer.highlights}
        docs = [(Document(page_content=result.pop(FIELDS_CONTENT), metadata={**(json.loads(result[FIELDS_METADATA]) if FIELDS_METADATA in result else {k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR}), **{'captions': {'text': result.get('@search.captions', [{}])[0].text, 'highlights': result.get('@search.captions', [{}])[0].highlights} if result.get('@search.captions') else {}, 'answers': semantic_answers_dict.get(result.get(FIELDS_ID, ''), '')}}), float(result['@search.score']), float(result['@search.reranker_score'])) for result in results]
        return docs

    @classmethod
    def from_texts(cls: Type[AzureSearch], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, azure_search_endpoint: str='', azure_search_key: str='', index_name: str='langchain-index', fields: Optional[List[SearchField]]=None, **kwargs: Any) -> AzureSearch:
        azure_search = cls(azure_search_endpoint, azure_search_key, index_name, embedding, fields=fields)
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search

    def as_retriever(self, **kwargs: Any) -> AzureSearchVectorStoreRetriever:
        """Return AzureSearchVectorStoreRetriever initialized from this VectorStore.

        Args:
            search_type (Optional[str]): Defines the type of search that
                the Retriever should perform.
                Can be "similarity" (default), "hybrid", or
                    "semantic_hybrid".
            search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                search function. Can include things like:
                    k: Amount of documents to return (Default: 4)
                    score_threshold: Minimum relevance threshold
                        for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                    lambda_mult: Diversity of results returned by MMR;
                        1 for minimum diversity and 0 for maximum. (Default: 0.5)
                    filter: Filter by document metadata

        Returns:
            AzureSearchVectorStoreRetriever: Retriever class for VectorStore.
        """
        tags = kwargs.pop('tags', None) or []
        tags.extend(self._get_retriever_tags())
        return AzureSearchVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)