from typing import Any, Dict, List, cast
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
class LlamaIndexRetriever(BaseRetriever):
    """`LlamaIndex` retriever.

    It is used for the question-answering with sources over
    an LlamaIndex data structure."""
    index: Any
    'LlamaIndex index to query.'
    query_kwargs: Dict = Field(default_factory=dict)
    'Keyword arguments to pass to the query method.'

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.core.base.response.schema import Response
            from llama_index.core.indices.base import BaseGPTIndex
        except ImportError:
            raise ImportError('You need to install `pip install llama-index` to use this retriever.')
        index = cast(BaseGPTIndex, self.index)
        response = index.query(query, **self.query_kwargs)
        response = cast(Response, response)
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.metadata or {}
            docs.append(Document(page_content=source_node.get_content(), metadata=metadata))
        return docs