from __future__ import annotations
import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
def _split_sources(self, answer: str) -> Tuple[str, str]:
    """Split sources from answer."""
    if re.search('SOURCES?:', answer, re.IGNORECASE):
        answer, sources = re.split('SOURCES?:|QUESTION:\\s', answer, flags=re.IGNORECASE)[:2]
        sources = re.split('\\n', sources)[0].strip()
    else:
        sources = ''
    return (answer, sources)