from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class VDMS(VectorStore):
    """Intel Lab's VDMS for vector-store workloads.

    To use, you should have both:
    - the ``vdms`` python package installed
    - a host (str) and port (int) associated with a deployed VDMS Server

    Visit https://github.com/IntelLabs/vdms/wiki more information.

    IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        client: VDMS Client used to connect to VDMS server
        collection_name: Name of data collection [Default: langchain]
        distance_strategy: Method used to calculate distances. VDMS supports
            "L2" (euclidean distance) or "IP" (inner product) [Default: L2]
        engine: Underlying implementation for indexing and computing distances.
            VDMS supports TileDBDense, TileDBSparse, FaissFlat, FaissIVFFlat,
            and Flinng [Default: FaissFlat]
        embedding: Any embedding function implementing
            `langchain_core.embeddings.Embeddings` interface.
        relevance_score_fn: Function for obtaining relevance score

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

            vectorstore = VDMS(
                client=VDMS_Client("localhost", 55555),
                embedding=HuggingFaceEmbeddings(),
                collection_name="langchain-demo",
                distance_strategy="L2",
                engine="FaissFlat",
            )
    """

    def __init__(self, client: vdms.vdms, *, embedding: Optional[Embeddings]=None, collection_name: str=DEFAULT_COLLECTION_NAME, distance_strategy: DISTANCE_METRICS='L2', engine: ENGINES='FaissFlat', relevance_score_fn: Optional[Callable[[float], float]]=None) -> None:
        self._client = client
        self.similarity_search_engine = engine
        self.distance_strategy = distance_strategy
        self.embedding = embedding
        self._check_required_inputs(collection_name)
        self.override_relevance_score_fn = relevance_score_fn
        self._collection_name = self.__add_set(collection_name, engine=self.similarity_search_engine, metric=self.distance_strategy)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding, Embeddings):
            return self.embedding.embed_documents(texts)
        else:
            p_str = 'Must provide `embedding` which is expected'
            p_str += ' to be an Embeddings object'
            raise ValueError(p_str)

    def _embed_image(self, uris: List[str]) -> List[List[float]]:
        if self.embedding is not None and hasattr(self.embedding, 'embed_image'):
            return self.embedding.embed_image(uris=uris)
        else:
            raise ValueError('Must provide `embedding` which has attribute `embed_image`')

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding, Embeddings):
            return self.embedding.embed_query(text)
        else:
            raise ValueError('Must provide `embedding` which is expected to be an Embeddings object')

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn
        if self.distance_strategy.lower() in ['ip', 'l2']:
            return lambda x: x
        else:
            raise ValueError(f'No supported normalization function for distance_strategy of {self.distance_strategy}.Consider providing relevance_score_fn to VDMS constructor.')

    def _similarity_search_with_relevance_scores(self, query: str, k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Optional[Dict[str, Any]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        if self.override_relevance_score_fn is None:
            kwargs['normalize_distance'] = True
        docs_and_scores = self.similarity_search_with_score(query, k, fetch_k, filter, **kwargs)
        docs_and_rel_scores: List[Any] = []
        for doc, score in docs_and_scores:
            if self.override_relevance_score_fn is None:
                docs_and_rel_scores.append((doc, score))
            else:
                docs_and_rel_scores.append((doc, self.override_relevance_score_fn(score)))
        return docs_and_rel_scores

    def __add(self, collection_name: str, texts: List[str], embeddings: List[List[float]], metadatas: Optional[Union[List[None], List[Dict[str, Any]]]]=None, ids: Optional[List[str]]=None) -> List:
        _len_check_if_sized(texts, embeddings, 'texts', 'embeddings')
        metadatas = metadatas if metadatas is not None else [None for _ in texts]
        _len_check_if_sized(texts, metadatas, 'texts', 'metadatas')
        ids = ids if ids is not None else [str(uuid.uuid4()) for _ in texts]
        _len_check_if_sized(texts, ids, 'texts', 'ids')
        all_queries: List[Any] = []
        all_blobs: List[Any] = []
        inserted_ids: List[Any] = []
        for meta, emb, doc, id in zip(metadatas, embeddings, texts, ids):
            query, blob = self.__get_add_query(collection_name, metadata=meta, embedding=emb, document=doc, id=id)
            if blob is not None:
                all_queries.append(query)
                all_blobs.append(blob)
                inserted_ids.append(id)
        response, response_array = self.__run_vdms_query(all_queries, all_blobs)
        return inserted_ids

    def __add_set(self, collection_name: str, engine: ENGINES='FaissFlat', metric: DISTANCE_METRICS='L2') -> str:
        query = _add_descriptorset('AddDescriptorSet', collection_name, self.embedding_dimension, engine=getattr(engine, 'value', engine), metric=getattr(metric, 'value', metric))
        response, _ = self.__run_vdms_query([query])
        if 'FailedCommand' in response[0]:
            raise ValueError(f'Failed to add collection {collection_name}')
        return collection_name

    def __delete(self, collection_name: str, ids: Union[None, List[str]]=None, constraints: Union[None, Dict[str, Any]]=None) -> bool:
        """
        Deletes entire collection if id is not provided
        """
        all_queries: List[Any] = []
        all_blobs: List[Any] = []
        collection_properties = self.__get_properties(collection_name)
        results = {'list': collection_properties}
        if constraints is None:
            constraints = {'_deletion': ['==', 1]}
        else:
            constraints['_deletion'] = ['==', 1]
        if ids is not None:
            constraints['id'] = ['==', ids[0]]
        query = _add_descriptor('FindDescriptor', collection_name, label=None, ref=None, props=None, link=None, k_neighbors=None, constraints=constraints, results=results)
        all_queries.append(query)
        response, response_array = self.__run_vdms_query(all_queries, all_blobs)
        return 'FindDescriptor' in response[0]

    def __get_add_query(self, collection_name: str, metadata: Optional[Any]=None, embedding: Union[List[float], None]=None, document: Optional[Any]=None, id: Optional[str]=None) -> Tuple[Dict[str, Dict[str, Any]], Union[bytes, None]]:
        if id is None:
            props: Dict[str, Any] = {}
        else:
            props = {'id': id}
            id_exists, query = _check_descriptor_exists_by_id(self._client, collection_name, id)
            if id_exists:
                skipped_value = {prop_key: prop_val[-1] for prop_key, prop_val in query['FindDescriptor']['constraints'].items()}
                pstr = f'[!] Embedding with id ({id}) exists in DB;'
                pstr += 'Therefore, skipped and not inserted'
                print(pstr)
                print(f'\tSkipped values are: {skipped_value}')
                return (query, None)
        if metadata:
            props.update(metadata)
        if document:
            props['content'] = document
        for k in props.keys():
            if k not in self.collection_properties:
                self.collection_properties.append(k)
        query = _add_descriptor('AddDescriptor', collection_name, label=None, ref=None, props=props, link=None, k_neighbors=None, constraints=None, results=None)
        blob = embedding2bytes(embedding)
        return (query, blob)

    def __get_properties(self, collection_name: str, unique_entity: Optional[bool]=False, deletion: Optional[bool]=False) -> List[str]:
        find_query = _find_property_entity(collection_name, unique_entity=unique_entity, deletion=deletion)
        response, response_blob = self.__run_vdms_query([find_query])
        if len(response_blob) > 0:
            collection_properties = _bytes2str(response_blob[0]).split(',')
        else:
            collection_properties = deepcopy(DEFAULT_PROPERTIES)
        return collection_properties

    def __run_vdms_query(self, all_queries: List[Dict], all_blobs: Optional[List]=[], print_last_response: Optional[bool]=False) -> Tuple[Any, Any]:
        response, response_array = self._client.query(all_queries, all_blobs)
        _ = _check_valid_response(all_queries, response)
        if print_last_response:
            self._client.print_last_response()
        return (response, response_array)

    def __update(self, collection_name: str, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[Union[List[None], List[Dict[str, Any]]]]=None) -> None:
        """
        Updates (find, delete, add) a collection based on id.
        If more than one collection returned with id, error occuers
        """
        _len_check_if_sized(ids, documents, 'ids', 'documents')
        _len_check_if_sized(ids, embeddings, 'ids', 'embeddings')
        metadatas = metadatas if metadatas is not None else [None for _ in ids]
        _len_check_if_sized(ids, metadatas, 'ids', 'metadatas')
        orig_props = self.__get_properties(collection_name)
        updated_ids: List[Any] = []
        for meta, emb, doc, id in zip(metadatas, embeddings, documents, ids):
            results = {'list': self.collection_properties}
            constraints = {'_deletion': ['==', 1]}
            if id is not None:
                constraints['id'] = ['==', id]
            query = _add_descriptor('FindDescriptor', collection_name, label=None, ref=None, props=None, link=None, k_neighbors=None, constraints=constraints, results=results)
            response, response_array = self.__run_vdms_query([query])
            query, blob = self.__get_add_query(collection_name, metadata=meta, embedding=emb, document=doc, id=id)
            if blob is not None:
                response, response_array = self.__run_vdms_query([query], [blob])
                updated_ids.append(id)
        self.__update_properties(collection_name, orig_props, self.collection_properties)

    def __update_properties(self, collection_name: str, current_collection_properties: List, new_collection_properties: Optional[List]) -> None:
        if new_collection_properties is not None:
            old_collection_properties = deepcopy(current_collection_properties)
            for prop in new_collection_properties:
                if prop not in current_collection_properties:
                    current_collection_properties.append(prop)
            if current_collection_properties != old_collection_properties:
                all_queries, blob_arr = _build_property_query(collection_name, command_type='update', all_properties=current_collection_properties)
                response, _ = self.__run_vdms_query(all_queries, [blob_arr])

    def add_images(self, uris: List[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, batch_size: int=DEFAULT_INSERT_BATCH_SIZE, add_path: Optional[bool]=True, **kwargs: Any) -> List[str]:
        """Run more images through the embeddings and add to the vectorstore.

        Images are added as embeddings (AddDescriptor) instead of separate
        entity (AddImage) within VDMS to leverage similarity search capability

        Args:
            uris: List of paths to the images to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            batch_size (int): Number of concurrent requests to send to the server.
            add_path: Bool to add image path as metadata

        Returns:
            List of ids from adding images into the vectorstore.
        """
        b64_texts = [self.encode_image(image_path=uri) for uri in uris]
        if add_path and metadatas:
            for midx, uri in enumerate(uris):
                metadatas[midx]['image_path'] = uri
        elif add_path:
            metadatas = []
            for uri in uris:
                metadatas.append({'image_path': uri})
        ids = ids if ids is not None else [str(uuid.uuid4()) for _ in uris]
        embeddings = self._embed_image(uris=uris)
        if metadatas is None:
            metadatas = [{} for _ in uris]
        else:
            metadatas = [_validate_vdms_properties(m) for m in metadatas]
        self.__from(texts=b64_texts, embeddings=embeddings, ids=ids, metadatas=metadatas, batch_size=batch_size, **kwargs)
        return ids

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, batch_size: int=DEFAULT_INSERT_BATCH_SIZE, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            batch_size (int): Number of concurrent requests to send to the server.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embed_documents(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        else:
            metadatas = [_validate_vdms_properties(m) for m in metadatas]
        inserted_ids = self.__from(texts=texts, embeddings=embeddings, ids=ids, metadatas=metadatas, batch_size=batch_size, **kwargs)
        return inserted_ids

    def __from(self, texts: List[str], embeddings: List[List[float]], ids: List[str], metadatas: Optional[List[dict]]=None, batch_size: int=DEFAULT_INSERT_BATCH_SIZE, **kwargs: Any) -> List[str]:
        orig_props = self.__get_properties(self._collection_name)
        inserted_ids: List[str] = []
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_embedding_vectors = embeddings[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            if metadatas:
                batch_metadatas = metadatas[start_idx:end_idx]
            result = self.__add(self._collection_name, embeddings=batch_embedding_vectors, texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
            inserted_ids.extend(result)
        self.__update_properties(self._collection_name, orig_props, self.collection_properties)
        return inserted_ids

    def _check_required_inputs(self, collection_name: str) -> None:
        if not self._client.is_connected():
            raise ValueError('VDMS client must be connected to a VDMS server.' + 'Please use VDMS_Client to establish a connection')
        if self.distance_strategy not in AVAILABLE_DISTANCE_METRICS:
            raise ValueError("distance_strategy must be either 'L2' or 'IP'")
        if self.similarity_search_engine not in AVAILABLE_ENGINES:
            raise ValueError("engine must be either 'TileDBDense', 'TileDBSparse', " + "'FaissFlat', 'FaissIVFFlat', or 'Flinng'")
        if self.embedding is None:
            raise ValueError('Must provide embedding function')
        self.embedding_dimension = len(self._embed_query('This is a sample sentence.'))
        current_props = self.__get_properties(collection_name)
        if hasattr(self, 'collection_properties'):
            self.collection_properties.extend(current_props)
        else:
            self.collection_properties: List[str] = current_props

    def count(self, collection_name: str) -> int:
        all_queries: List[Any] = []
        all_blobs: List[Any] = []
        results = {'count': '', 'list': ['id']}
        query = _add_descriptor('FindDescriptor', collection_name, label=None, ref=None, props=None, link=None, k_neighbors=None, constraints=None, results=results)
        all_queries.append(query)
        response, response_array = self.__run_vdms_query(all_queries, all_blobs)
        return response[0]['FindDescriptor']['returned']

    def decode_image(self, base64_image: str) -> bytes:
        return base64.b64decode(base64_image)

    def delete(self, ids: Optional[List[str]]=None, collection_name: Optional[str]=None, constraints: Optional[Dict]=None, **kwargs: Any) -> bool:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        name = collection_name if collection_name is not None else self._collection_name
        return self.__delete(name, ids=ids, constraints=constraints)

    def get_k_candidates(self, setname: str, fetch_k: Optional[int], results: Optional[Dict[str, Any]]=None, all_blobs: Optional[List]=None, normalize: Optional[bool]=False) -> Tuple[List[Dict[str, Any]], List, float]:
        max_dist = 1
        command_str = 'FindDescriptor'
        query = _add_descriptor(command_str, setname, k_neighbors=fetch_k, results=results)
        response, response_array = self.__run_vdms_query([query], all_blobs)
        if normalize:
            max_dist = response[0][command_str]['entities'][-1]['_distance']
        return (response, response_array, max_dist)

    def get_descriptor_response(self, command_str: str, setname: str, k_neighbors: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, constraints: Optional[dict]=None, results: Optional[Dict[str, Any]]=None, query_embedding: Optional[List[float]]=None, normalize_distance: bool=False) -> Tuple[List[Dict[str, Any]], List]:
        all_blobs: List[Any] = []
        blob = embedding2bytes(query_embedding)
        if blob is not None:
            all_blobs.append(blob)
        if constraints is None:
            response, response_array, max_dist = self.get_k_candidates(setname, k_neighbors, results, all_blobs, normalize=normalize_distance)
        else:
            if results is None:
                results = {'list': ['id']}
            elif 'list' not in results:
                results['list'] = ['id']
            elif 'id' not in results['list']:
                results['list'].append('id')
            query = _add_descriptor(command_str, setname, constraints=constraints, results=results)
            response, response_array = self.__run_vdms_query([query])
            ids_of_interest = [ent['id'] for ent in response[0][command_str]['entities']]
            response, response_array, max_dist = self.get_k_candidates(setname, fetch_k, results, all_blobs, normalize=normalize_distance)
            new_entities: List[Dict] = []
            for ent in response[0][command_str]['entities']:
                if ent['id'] in ids_of_interest:
                    new_entities.append(ent)
                if len(new_entities) == k_neighbors:
                    break
            response[0][command_str]['entities'] = new_entities
            response[0][command_str]['returned'] = len(new_entities)
            if len(new_entities) < k_neighbors:
                p_str = 'Returned items < k_neighbors; Try increasing fetch_k'
                print(p_str)
        if normalize_distance:
            max_dist = 1.0 if max_dist == 0 else max_dist
            for ent_idx, ent in enumerate(response[0][command_str]['entities']):
                ent['_distance'] = ent['_distance'] / max_dist
                response[0][command_str]['entities'][ent_idx]['_distance'] = ent['_distance']
        return (response, response_array)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            blob = f.read()
            return base64.b64encode(blob).decode('utf-8')

    @classmethod
    def from_documents(cls: Type[VDMS], documents: List[Document], embedding: Optional[Embeddings]=None, ids: Optional[List[str]]=None, batch_size: int=DEFAULT_INSERT_BATCH_SIZE, collection_name: str=DEFAULT_COLLECTION_NAME, **kwargs: Any) -> VDMS:
        """Create a VDMS vectorstore from a list of documents.

        Args:
            collection_name (str): Name of the collection to create.
            documents (List[Document]): List of documents to add to vectorstore.
            embedding (Embeddings): Embedding function. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            batch_size (int): Number of concurrent requests to send to the server.

        Returns:
            VDMS: VDMS vectorstore.
        """
        client: vdms.vdms = kwargs['client']
        return cls.from_texts(client=client, texts=[doc.page_content for doc in documents], metadatas=[doc.metadata for doc in documents], embedding=embedding, ids=ids, batch_size=batch_size, collection_name=collection_name)

    @classmethod
    def from_texts(cls: Type[VDMS], texts: List[str], embedding: Optional[Embeddings]=None, metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, batch_size: int=DEFAULT_INSERT_BATCH_SIZE, collection_name: str=DEFAULT_COLLECTION_NAME, **kwargs: Any) -> VDMS:
        """Create a VDMS vectorstore from a raw documents.

        Args:
            texts (List[str]): List of texts to add to the collection.
            embedding (Embeddings): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            batch_size (int): Number of concurrent requests to send to the server.
            collection_name (str): Name of the collection to create.

        Returns:
            VDMS: VDMS vectorstore.
        """
        client: vdms.vdms = kwargs['client']
        vdms_collection = cls(collection_name=collection_name, embedding=embedding, client=client)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        vdms_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids, batch_size=batch_size)
        return vdms_collection

    def get(self, collection_name: str, constraints: Optional[Dict]=None, limit: Optional[int]=None, include: List[str]=['metadata']) -> Tuple[Any, Any]:
        """Gets the collection.
        Get embeddings and their associated data from the data store.
        If no constraints provided returns all embeddings up to limit.

        Args:
            constraints: A dict used to filter results by.
                   E.g. `{"color" : ["==", "red"], "price": [">", 4.00]}`. Optional.
            limit: The number of documents to return. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.
        """
        all_queries: List[Any] = []
        all_blobs: List[Any] = []
        results: Dict[str, Any] = {'count': ''}
        if limit is not None:
            results['limit'] = limit
        if 'metadata' in include:
            collection_properties = self.__get_properties(collection_name)
            results['list'] = collection_properties
        if 'embeddings' in include:
            results['blob'] = True
        query = _add_descriptor('FindDescriptor', collection_name, k_neighbors=None, constraints=constraints, results=results)
        all_queries.append(query)
        response, response_array = self.__run_vdms_query(all_queries, all_blobs)
        return (response, response_array)

    def max_marginal_relevance_search(self, query: str, k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, lambda_mult: float=0.5, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self.embedding is None:
            raise ValueError('For MMR search, you must specify an embedding function oncreation.')
        embedding_vector: List[float] = self._embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(embedding_vector, k, fetch_k, lambda_mult=lambda_mult, filter=filter)
        return docs

    def max_marginal_relevance_search_by_vector(self, embedding: List[float], k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, lambda_mult: float=0.5, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self.query_collection_embeddings(query_embeddings=[embedding], n_results=fetch_k, filter=filter, include=['metadatas', 'documents', 'distances', 'embeddings'])
        embedding_list = [list(_bytes2embedding(result)) for result in results[0][1]]
        mmr_selected = maximal_marginal_relevance(np.array(embedding, dtype=np.float32), embedding_list, k=k, lambda_mult=lambda_mult)
        candidates = _results_to_docs(results)
        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search_with_score(self, query: str, k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, lambda_mult: float=0.5, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self.embedding is None:
            raise ValueError('For MMR search, you must specify an embedding function oncreation.')
        embedding = self._embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(embedding, k, fetch_k, lambda_mult=lambda_mult, filter=filter)
        return docs

    def max_marginal_relevance_search_with_score_by_vector(self, embedding: List[float], k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, lambda_mult: float=0.5, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self.query_collection_embeddings(query_embeddings=[embedding], n_results=fetch_k, filter=filter, include=['metadatas', 'documents', 'distances', 'embeddings'])
        embedding_list = [list(_bytes2embedding(result)) for result in results[0][1]]
        mmr_selected = maximal_marginal_relevance(np.array(embedding, dtype=np.float32), embedding_list, k=k, lambda_mult=lambda_mult)
        candidates = _results_to_docs_and_scores(results)
        selected_results = [(r, s) for i, (r, s) in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def query_collection_embeddings(self, query_embeddings: Optional[List[List[float]]]=None, collection_name: Optional[str]=None, n_results: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Union[None, Dict[str, Any]]=None, results: Union[None, Dict[str, Any]]=None, normalize_distance: bool=False, **kwargs: Any) -> List[Tuple[Dict[str, Any], List]]:
        all_responses: List[Any] = []
        if collection_name is None:
            collection_name = self._collection_name
        if query_embeddings is None:
            return all_responses
        include = kwargs.get('include', ['metadatas'])
        if results is None and 'metadatas' in include:
            results = {'list': self.collection_properties, 'blob': 'embeddings' in include}
        for qemb in query_embeddings:
            response, response_array = self.get_descriptor_response('FindDescriptor', collection_name, k_neighbors=n_results, fetch_k=fetch_k, constraints=filter, results=results, normalize_distance=normalize_distance, query_embedding=qemb)
            all_responses.append([response, response_array])
        return all_responses

    def similarity_search(self, query: str, k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Document]:
        """Run similarity search with VDMS.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 3.
            fetch_k (int): Number of candidates to fetch for knn (>= k).
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, fetch_k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(self, embedding: List[float], k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 3.
            fetch_k (int): Number of candidates to fetch for knn (>= k).
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.query_collection_embeddings(query_embeddings=[embedding], n_results=k, fetch_k=fetch_k, filter=filter, **kwargs)
        return _results_to_docs(results)

    def similarity_search_with_score(self, query: str, k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Run similarity search with VDMS with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 3.
            fetch_k (int): Number of candidates to fetch for knn (>= k).
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        if self.embedding is None:
            raise ValueError('Must provide embedding function')
        else:
            query_embedding: List[float] = self._embed_query(query)
            results = self.query_collection_embeddings(query_embeddings=[query_embedding], n_results=k, fetch_k=fetch_k, filter=filter, **kwargs)
        return _results_to_docs_and_scores(results)

    def similarity_search_with_score_by_vector(self, embedding: List[float], k: int=DEFAULT_K, fetch_k: int=DEFAULT_FETCH_K, filter: Optional[Dict[str, List]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """
        Return docs most similar to embedding vector and similarity score.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 3.
            fetch_k (int): Number of candidates to fetch for knn (>= k).
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        kwargs['normalize_distance'] = True
        results = self.query_collection_embeddings(query_embeddings=[embedding], n_results=k, fetch_k=fetch_k, filter=filter, **kwargs)
        return _results_to_docs_and_scores(results)

    def update_document(self, collection_name: str, document_id: str, document: Document) -> None:
        """Update a document in the collection.

        Args:
            document_id (str): ID of the document to update.
            document (Document): Document to update.
        """
        return self.update_documents(collection_name, [document_id], [document])

    def update_documents(self, collection_name: str, ids: List[str], documents: List[Document]) -> None:
        """Update a document in the collection.

        Args:
            ids (List[str]): List of ids of the document to update.
            documents (List[Document]): List of documents to update.
        """
        text = [document.page_content for document in documents]
        metadata = [_validate_vdms_properties(document.metadata) for document in documents]
        embeddings = self._embed_documents(text)
        self.__update(collection_name, ids, metadatas=metadata, embeddings=embeddings, documents=text)