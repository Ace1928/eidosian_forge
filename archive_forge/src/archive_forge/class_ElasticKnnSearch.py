from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@deprecated('0.0.1', alternative='Use ElasticsearchStore class in langchain-elasticsearch package', pending=True)
class ElasticKnnSearch(VectorStore):
    """[DEPRECATED] `Elasticsearch` with k-nearest neighbor search
    (`k-NN`) vector store.

    Recommended to use ElasticsearchStore instead, which supports
    metadata filtering, customising the query retriever and much more!

    You can read more on ElasticsearchStore:
    https://python.langchain.com/docs/integrations/vectorstores/elasticsearch

    It creates an Elasticsearch index of text data that
    can be searched using k-NN search. The text data is transformed into
    vector embeddings using a provided embedding model, and these embeddings
    are stored in the Elasticsearch index.

    Attributes:
        index_name (str): The name of the Elasticsearch index.
        embedding (Embeddings): The embedding model to use for transforming text data
            into vector embeddings.
        es_connection (Elasticsearch, optional): An existing Elasticsearch connection.
        es_cloud_id (str, optional): The Cloud ID of your Elasticsearch Service
            deployment.
        es_user (str, optional): The username for your Elasticsearch Service deployment.
        es_password (str, optional): The password for your Elasticsearch Service
            deployment.
        vector_query_field (str, optional): The name of the field in the Elasticsearch
            index that contains the vector embeddings.
        query_field (str, optional): The name of the field in the Elasticsearch index
            that contains the original text data.

    Usage:
        >>> from embeddings import Embeddings
        >>> embedding = Embeddings.load('glove')
        >>> es_search = ElasticKnnSearch('my_index', embedding)
        >>> es_search.add_texts(['Hello world!', 'Another text'])
        >>> results = es_search.knn_search('Hello')
        [(Document(page_content='Hello world!', metadata={}), 0.9)]
    """

    def __init__(self, index_name: str, embedding: Embeddings, es_connection: Optional['Elasticsearch']=None, es_cloud_id: Optional[str]=None, es_user: Optional[str]=None, es_password: Optional[str]=None, vector_query_field: Optional[str]='vector', query_field: Optional[str]='text'):
        try:
            import elasticsearch
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. Please install it with `pip install elasticsearch`.')
        warnings.warn('ElasticKnnSearch will be removed in a future release.Use ElasticsearchStore instead. See Elasticsearch integration docs on how to upgrade.')
        self.embedding = embedding
        self.index_name = index_name
        self.query_field = query_field
        self.vector_query_field = vector_query_field
        if es_connection is not None:
            self.client = es_connection
        elif es_cloud_id and es_user and es_password:
            self.client = elasticsearch.Elasticsearch(cloud_id=es_cloud_id, basic_auth=(es_user, es_password))
        else:
            raise ValueError('Either provide a pre-existing Elasticsearch connection,                 or valid credentials for creating a new connection.')

    @staticmethod
    def _default_knn_mapping(dims: int, similarity: Optional[str]='dot_product') -> Dict:
        return {'properties': {'text': {'type': 'text'}, 'vector': {'type': 'dense_vector', 'dims': dims, 'index': True, 'similarity': similarity}}}

    def _default_knn_query(self, query_vector: Optional[List[float]]=None, query: Optional[str]=None, model_id: Optional[str]=None, k: Optional[int]=10, num_candidates: Optional[int]=10) -> Dict:
        knn: Dict = {'field': self.vector_query_field, 'k': k, 'num_candidates': num_candidates}
        if query_vector and (not model_id):
            knn['query_vector'] = query_vector
        elif query and model_id:
            knn['query_vector_builder'] = {'text_embedding': {'model_id': model_id, 'model_text': query}}
        else:
            raise ValueError('Either `query_vector` or `model_id` must be provided, but not both.')
        return knn

    def similarity_search(self, query: str, k: int=4, filter: Optional[dict]=None, **kwargs: Any) -> List[Document]:
        """
        Pass through to `knn_search`
        """
        results = self.knn_search(query=query, k=k, **kwargs)
        return [doc for doc, score in results]

    def similarity_search_with_score(self, query: str, k: int=10, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Pass through to `knn_search including score`"""
        return self.knn_search(query=query, k=k, **kwargs)

    def knn_search(self, query: Optional[str]=None, k: Optional[int]=10, query_vector: Optional[List[float]]=None, model_id: Optional[str]=None, size: Optional[int]=10, source: Optional[bool]=True, fields: Optional[Union[List[Mapping[str, Any]], Tuple[Mapping[str, Any], ...], None]]=None, page_content: Optional[str]='text') -> List[Tuple[Document, float]]:
        """
        Perform a k-NN search on the Elasticsearch index.

        Args:
            query (str, optional): The query text to search for.
            k (int, optional): The number of nearest neighbors to return.
            query_vector (List[float], optional): The query vector to search for.
            model_id (str, optional): The ID of the model to use for transforming the
                query text into a vector.
            size (int, optional): The number of search results to return.
            source (bool, optional): Whether to return the source of the search results.
            fields (List[Mapping[str, Any]], optional): The fields to return in the
                search results.
            page_content (str, optional): The name of the field that contains the page
                content.

        Returns:
            A list of tuples, where each tuple contains a Document object and a score.
        """
        if not source and (fields is None or not any((page_content in field for field in fields))):
            raise ValueError('If source=False `page_content` field must be in `fields`')
        knn_query_body = self._default_knn_query(query_vector=query_vector, query=query, model_id=model_id, k=k)
        response = self.client.search(index=self.index_name, knn=knn_query_body, size=size, source=source, fields=fields)
        hits = [hit for hit in response['hits']['hits']]
        docs_and_scores = [(Document(page_content=hit['_source'][page_content] if source else hit['fields'][page_content][0], metadata=hit['fields'] if fields else {}), hit['_score']) for hit in hits]
        return docs_and_scores

    def knn_hybrid_search(self, query: Optional[str]=None, k: Optional[int]=10, query_vector: Optional[List[float]]=None, model_id: Optional[str]=None, size: Optional[int]=10, source: Optional[bool]=True, knn_boost: Optional[float]=0.9, query_boost: Optional[float]=0.1, fields: Optional[Union[List[Mapping[str, Any]], Tuple[Mapping[str, Any], ...], None]]=None, page_content: Optional[str]='text') -> List[Tuple[Document, float]]:
        """
        Perform a hybrid k-NN and text search on the Elasticsearch index.

        Args:
            query (str, optional): The query text to search for.
            k (int, optional): The number of nearest neighbors to return.
            query_vector (List[float], optional): The query vector to search for.
            model_id (str, optional): The ID of the model to use for transforming the
                query text into a vector.
            size (int, optional): The number of search results to return.
            source (bool, optional): Whether to return the source of the search results.
            knn_boost (float, optional): The boost value to apply to the k-NN search
                results.
            query_boost (float, optional): The boost value to apply to the text search
                results.
            fields (List[Mapping[str, Any]], optional): The fields to return in the
                search results.
            page_content (str, optional): The name of the field that contains the page
                content.

        Returns:
            A list of tuples, where each tuple contains a Document object and a score.
        """
        if not source and (fields is None or not any((page_content in field for field in fields))):
            raise ValueError('If source=False `page_content` field must be in `fields`')
        knn_query_body = self._default_knn_query(query_vector=query_vector, query=query, model_id=model_id, k=k)
        knn_query_body['boost'] = knn_boost
        match_query_body = {'match': {self.query_field: {'query': query, 'boost': query_boost}}}
        response = self.client.search(index=self.index_name, query=match_query_body, knn=knn_query_body, fields=fields, size=size, source=source)
        hits = [hit for hit in response['hits']['hits']]
        docs_and_scores = [(Document(page_content=hit['_source'][page_content] if source else hit['fields'][page_content][0], metadata=hit['fields'] if fields else {}), hit['_score']) for hit in hits]
        return docs_and_scores

    def create_knn_index(self, mapping: Dict) -> None:
        """
        Create a new k-NN index in Elasticsearch.

        Args:
            mapping (Dict): The mapping to use for the new index.

        Returns:
            None
        """
        self.client.indices.create(index=self.index_name, mappings=mapping)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[Dict[Any, Any]]]=None, model_id: Optional[str]=None, refresh_indices: bool=False, **kwargs: Any) -> List[str]:
        """
        Add a list of texts to the Elasticsearch index.

        Args:
            texts (Iterable[str]): The texts to add to the index.
            metadatas (List[Dict[Any, Any]], optional): A list of metadata dictionaries
                to associate with the texts.
            model_id (str, optional): The ID of the model to use for transforming the
                texts into vectors.
            refresh_indices (bool, optional): Whether to refresh the Elasticsearch
                indices after adding the texts.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A list of IDs for the added texts.
        """
        if not self.client.indices.exists(index=self.index_name):
            dims = kwargs.get('dims')
            if dims is None:
                raise ValueError("ElasticKnnSearch requires 'dims' parameter")
            similarity = kwargs.get('similarity')
            optional_args = {}
            if similarity is not None:
                optional_args['similarity'] = similarity
            mapping = self._default_knn_mapping(dims=dims, **optional_args)
            self.create_knn_index(mapping)
        embeddings = self.embedding.embed_documents(list(texts))
        body: List[Mapping[str, Any]] = []
        for text, vector in zip(texts, embeddings):
            body.extend([{'index': {'_index': self.index_name}}, {'text': text, 'vector': vector}])
        responses = self.client.bulk(operations=body)
        ids = [item['index']['_id'] for item in responses['items'] if item['index']['result'] == 'created']
        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[Dict[Any, Any]]]=None, **kwargs: Any) -> ElasticKnnSearch:
        """
        Create a new ElasticKnnSearch instance and add a list of texts to the
            Elasticsearch index.

        Args:
            texts (List[str]): The texts to add to the index.
            embedding (Embeddings): The embedding model to use for transforming the
                texts into vectors.
            metadatas (List[Dict[Any, Any]], optional): A list of metadata dictionaries
                to associate with the texts.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A new ElasticKnnSearch instance.
        """
        index_name = kwargs.get('index_name', str(uuid.uuid4()))
        es_connection = kwargs.get('es_connection')
        es_cloud_id = kwargs.get('es_cloud_id')
        es_user = kwargs.get('es_user')
        es_password = kwargs.get('es_password')
        vector_query_field = kwargs.get('vector_query_field', 'vector')
        query_field = kwargs.get('query_field', 'text')
        model_id = kwargs.get('model_id')
        dims = kwargs.get('dims')
        if dims is None:
            raise ValueError("ElasticKnnSearch requires 'dims' parameter")
        optional_args = {}
        if vector_query_field is not None:
            optional_args['vector_query_field'] = vector_query_field
        if query_field is not None:
            optional_args['query_field'] = query_field
        knnvectorsearch = cls(index_name=index_name, embedding=embedding, es_connection=es_connection, es_cloud_id=es_cloud_id, es_user=es_user, es_password=es_password, **optional_args)
        knnvectorsearch.add_texts(texts, model_id=model_id, dims=dims, **optional_args)
        return knnvectorsearch