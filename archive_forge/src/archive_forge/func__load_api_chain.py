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
def _load_api_chain(config: dict, **kwargs: Any) -> APIChain:
    if 'api_request_chain' in config:
        api_request_chain_config = config.pop('api_request_chain')
        api_request_chain = load_chain_from_config(api_request_chain_config, **kwargs)
    elif 'api_request_chain_path' in config:
        api_request_chain = load_chain(config.pop('api_request_chain_path'))
    else:
        raise ValueError('One of `api_request_chain` or `api_request_chain_path` must be present.')
    if 'api_answer_chain' in config:
        api_answer_chain_config = config.pop('api_answer_chain')
        api_answer_chain = load_chain_from_config(api_answer_chain_config, **kwargs)
    elif 'api_answer_chain_path' in config:
        api_answer_chain = load_chain(config.pop('api_answer_chain_path'), **kwargs)
    else:
        raise ValueError('One of `api_answer_chain` or `api_answer_chain_path` must be present.')
    if 'requests_wrapper' in kwargs:
        requests_wrapper = kwargs.pop('requests_wrapper')
    else:
        raise ValueError('`requests_wrapper` must be present.')
    return APIChain(api_request_chain=api_request_chain, api_answer_chain=api_answer_chain, requests_wrapper=requests_wrapper, **config)