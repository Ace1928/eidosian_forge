from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForChainRun, Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra
from langchain_text_splitters import TextSplitter
from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
Return output key.

        :meta private:
        