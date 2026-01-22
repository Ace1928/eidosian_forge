from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
    num_docs = len(docs)
    if self.max_tokens_limit and isinstance(self.combine_docs_chain, StuffDocumentsChain):
        tokens = [self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content) for doc in docs]
        token_count = sum(tokens[:num_docs])
        while token_count > self.max_tokens_limit:
            num_docs -= 1
            token_count -= tokens[num_docs]
    return docs[:num_docs]