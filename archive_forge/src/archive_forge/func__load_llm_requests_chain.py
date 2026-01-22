import json
from pathlib import Path
from typing import Any, Union
import yaml
from langchain_community.llms.loading import load_llm, load_llm_from_config
from langchain_core.prompts.loading import (
from langchain.chains import ReduceDocumentsChain
from langchain.chains.api.base import APIChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.llm_requests import LLMRequestsChain
from langchain.chains.qa_with_sources.base import QAWithSourcesChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
def _load_llm_requests_chain(config: dict, **kwargs: Any) -> LLMRequestsChain:
    if 'llm_chain' in config:
        llm_chain_config = config.pop('llm_chain')
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif 'llm_chain_path' in config:
        llm_chain = load_chain(config.pop('llm_chain_path'), **kwargs)
    else:
        raise ValueError('One of `llm_chain` or `llm_chain_path` must be present.')
    if 'requests_wrapper' in kwargs:
        requests_wrapper = kwargs.pop('requests_wrapper')
        return LLMRequestsChain(llm_chain=llm_chain, requests_wrapper=requests_wrapper, **config)
    else:
        return LLMRequestsChain(llm_chain=llm_chain, **config)