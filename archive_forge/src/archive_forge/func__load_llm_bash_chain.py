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
def _load_llm_bash_chain(config: dict, **kwargs: Any) -> Any:
    from langchain_experimental.llm_bash.base import LLMBashChain
    llm_chain = None
    if 'llm_chain' in config:
        llm_chain_config = config.pop('llm_chain')
        llm_chain = load_chain_from_config(llm_chain_config, **kwargs)
    elif 'llm_chain_path' in config:
        llm_chain = load_chain(config.pop('llm_chain_path'), **kwargs)
    elif 'llm' in config:
        llm_config = config.pop('llm')
        llm = load_llm_from_config(llm_config, **kwargs)
    elif 'llm_path' in config:
        llm = load_llm(config.pop('llm_path'), **kwargs)
    else:
        raise ValueError('One of `llm_chain` or `llm_chain_path` must be present.')
    if 'prompt' in config:
        prompt_config = config.pop('prompt')
        prompt = load_prompt_from_config(prompt_config)
    elif 'prompt_path' in config:
        prompt = load_prompt(config.pop('prompt_path'))
    if llm_chain:
        return LLMBashChain(llm_chain=llm_chain, prompt=prompt, **config)
    else:
        return LLMBashChain(llm=llm, prompt=prompt, **config)