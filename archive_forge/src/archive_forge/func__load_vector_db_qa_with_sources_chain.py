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
def _load_vector_db_qa_with_sources_chain(config: dict, **kwargs: Any) -> VectorDBQAWithSourcesChain:
    if 'vectorstore' in kwargs:
        vectorstore = kwargs.pop('vectorstore')
    else:
        raise ValueError('`vectorstore` must be present.')
    if 'combine_documents_chain' in config:
        combine_documents_chain_config = config.pop('combine_documents_chain')
        combine_documents_chain = load_chain_from_config(combine_documents_chain_config, **kwargs)
    elif 'combine_documents_chain_path' in config:
        combine_documents_chain = load_chain(config.pop('combine_documents_chain_path'), **kwargs)
    else:
        raise ValueError('One of `combine_documents_chain` or `combine_documents_chain_path` must be present.')
    return VectorDBQAWithSourcesChain(combine_documents_chain=combine_documents_chain, vectorstore=vectorstore, **config)