from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
@deprecated(since='0.0.33', removal='0.2.0', alternative_import='langchain_google_community.VertexAISearchRetriever')
class GoogleVertexAISearchRetriever(BaseRetriever, _BaseGoogleVertexAISearchRetriever):
    """`Google Vertex AI Search` retriever.

    For a detailed explanation of the Vertex AI Search concepts
    and configuration parameters, refer to the product documentation.
    https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction
    """
    filter: Optional[str] = None
    'Filter expression.'
    get_extractive_answers: bool = False
    'If True return Extractive Answers, otherwise return Extractive Segments or Snippets.'
    max_documents: int = Field(default=5, ge=1, le=100)
    'The maximum number of documents to return.'
    max_extractive_answer_count: int = Field(default=1, ge=1, le=5)
    'The maximum number of extractive answers returned in each search result.\n    At most 5 answers will be returned for each SearchResult.\n    '
    max_extractive_segment_count: int = Field(default=1, ge=1, le=1)
    'The maximum number of extractive segments returned in each search result.\n    Currently one segment will be returned for each SearchResult.\n    '
    query_expansion_condition: int = Field(default=1, ge=0, le=2)
    'Specification to determine under which conditions query expansion should occur.\n    0 - Unspecified query expansion condition. In this case, server behavior defaults \n        to disabled\n    1 - Disabled query expansion. Only the exact search query is used, even if \n        SearchResponse.total_size is zero.\n    2 - Automatic query expansion built by the Search API.\n    '
    spell_correction_mode: int = Field(default=2, ge=0, le=2)
    'Specification to determine under which conditions query expansion should occur.\n    0 - Unspecified spell correction mode. In this case, server behavior defaults \n        to auto.\n    1 - Suggestion only. Search API will try to find a spell suggestion if there is any\n        and put in the `SearchResponse.corrected_query`.\n        The spell suggestion will not be used as the search query.\n    2 - Automatic spell correction built by the Search API.\n        Search will be based on the corrected query if found.\n    '
    _client: SearchServiceClient
    _serving_config: str

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.ignore
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **kwargs: Any) -> None:
        """Initializes private fields."""
        try:
            from google.cloud.discoveryengine_v1beta import SearchServiceClient
        except ImportError as exc:
            raise ImportError('google.cloud.discoveryengine is not installed.Please install it with pip install google-cloud-discoveryengine') from exc
        super().__init__(**kwargs)
        self._client = SearchServiceClient(credentials=self.credentials, client_options=self.client_options, client_info=get_client_info(module='vertex-ai-search'))
        if self.engine_data_type == 3 and (not self.search_engine_id):
            raise ValueError('search_engine_id must be specified for blended search apps.')
        if self.search_engine_id:
            self._serving_config = f'projects/{self.project_id}/locations/{self.location_id}/collections/default_collection/engines/{self.search_engine_id}/servingConfigs/default_config'
        elif self.data_store_id:
            self._serving_config = self._client.serving_config_path(project=self.project_id, location=self.location_id, data_store=self.data_store_id, serving_config=self.serving_config_id)
        else:
            raise ValueError('Either data_store_id or search_engine_id must be specified.')

    def _create_search_request(self, query: str) -> SearchRequest:
        """Prepares a SearchRequest object."""
        from google.cloud.discoveryengine_v1beta import SearchRequest
        query_expansion_spec = SearchRequest.QueryExpansionSpec(condition=self.query_expansion_condition)
        spell_correction_spec = SearchRequest.SpellCorrectionSpec(mode=self.spell_correction_mode)
        if self.engine_data_type == 0:
            if self.get_extractive_answers:
                extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_answer_count=self.max_extractive_answer_count)
            else:
                extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_segment_count=self.max_extractive_segment_count)
            content_search_spec = SearchRequest.ContentSearchSpec(extractive_content_spec=extractive_content_spec)
        elif self.engine_data_type == 1:
            content_search_spec = None
        elif self.engine_data_type in (2, 3):
            content_search_spec = SearchRequest.ContentSearchSpec(extractive_content_spec=SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_answer_count=self.max_extractive_answer_count), snippet_spec=SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True))
        else:
            raise NotImplementedError('Only data store type 0 (Unstructured), 1 (Structured),2 (Website), or 3 (Blended) are supported currently.' + f' Got {self.engine_data_type}')
        return SearchRequest(query=query, filter=self.filter, serving_config=self._serving_config, page_size=self.max_documents, content_search_spec=content_search_spec, query_expansion_spec=query_expansion_spec, spell_correction_spec=spell_correction_spec)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant for a query."""
        return self.get_relevant_documents_with_response(query)[0]

    def get_relevant_documents_with_response(self, query: str) -> Tuple[List[Document], Any]:
        from google.api_core.exceptions import InvalidArgument
        search_request = self._create_search_request(query)
        try:
            response = self._client.search(search_request)
        except InvalidArgument as exc:
            raise type(exc)(exc.message + ' This might be due to engine_data_type not set correctly.')
        if self.engine_data_type == 0:
            chunk_type = 'extractive_answers' if self.get_extractive_answers else 'extractive_segments'
            documents = self._convert_unstructured_search_response(response.results, chunk_type)
        elif self.engine_data_type == 1:
            documents = self._convert_structured_search_response(response.results)
        elif self.engine_data_type in (2, 3):
            chunk_type = 'extractive_answers' if self.get_extractive_answers else 'snippets'
            documents = self._convert_website_search_response(response.results, chunk_type)
        else:
            raise NotImplementedError('Only data store type 0 (Unstructured), 1 (Structured),2 (Website), or 3 (Blended) are supported currently.' + f' Got {self.engine_data_type}')
        return (documents, response)