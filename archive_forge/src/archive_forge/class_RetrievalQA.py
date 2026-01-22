from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
class RetrievalQA(BaseRetrievalQA):
    """Chain for question-answering against an index.

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAI
            from langchain.chains import RetrievalQA
            from langchain_community.vectorstores import FAISS
            from langchain_core.vectorstores import VectorStoreRetriever
            retriever = VectorStoreRetriever(vectorstore=FAISS(...))
            retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

    """
    retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(self, question: str, *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        return self.retriever.get_relevant_documents(question, callbacks=run_manager.get_child())

    async def _aget_docs(self, question: str, *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        return await self.retriever.aget_relevant_documents(question, callbacks=run_manager.get_child())

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return 'retrieval_qa'