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
def _load_refine_documents_chain(config: dict, **kwargs: Any) -> RefineDocumentsChain:
    if 'initial_llm_chain' in config:
        initial_llm_chain_config = config.pop('initial_llm_chain')
        initial_llm_chain = load_chain_from_config(initial_llm_chain_config, **kwargs)
    elif 'initial_llm_chain_path' in config:
        initial_llm_chain = load_chain(config.pop('initial_llm_chain_path'), **kwargs)
    else:
        raise ValueError('One of `initial_llm_chain` or `initial_llm_chain_path` must be present.')
    if 'refine_llm_chain' in config:
        refine_llm_chain_config = config.pop('refine_llm_chain')
        refine_llm_chain = load_chain_from_config(refine_llm_chain_config, **kwargs)
    elif 'refine_llm_chain_path' in config:
        refine_llm_chain = load_chain(config.pop('refine_llm_chain_path'), **kwargs)
    else:
        raise ValueError('One of `refine_llm_chain` or `refine_llm_chain_path` must be present.')
    if 'document_prompt' in config:
        prompt_config = config.pop('document_prompt')
        document_prompt = load_prompt_from_config(prompt_config)
    elif 'document_prompt_path' in config:
        document_prompt = load_prompt(config.pop('document_prompt_path'))
    return RefineDocumentsChain(initial_llm_chain=initial_llm_chain, refine_llm_chain=refine_llm_chain, document_prompt=document_prompt, **config)