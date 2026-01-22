from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
@deprecated(since='0.0.30', removal='0.2.0', alternative_import='langchain_cohere.CohereRagRetriever')
class CohereRagRetriever(BaseRetriever):
    """Cohere Chat API with RAG."""
    connectors: List[Dict] = Field(default_factory=lambda: [{'id': 'web-search'}])
    '\n    When specified, the model\'s reply will be enriched with information found by\n    querying each of the connectors (RAG). These will be returned as langchain\n    documents.\n\n    Currently only accepts {"id": "web-search"}.\n    '
    llm: BaseChatModel
    'Cohere ChatModel to use.'

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
        'Allow arbitrary types.'

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = self.llm.generate(messages, connectors=self.connectors, callbacks=run_manager.get_child(), **kwargs).generations[0][0]
        return _get_docs(res)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = (await self.llm.agenerate(messages, connectors=self.connectors, callbacks=run_manager.get_child(), **kwargs)).generations[0][0]
        return _get_docs(res)