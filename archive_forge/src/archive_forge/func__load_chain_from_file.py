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
def _load_chain_from_file(file: Union[str, Path], **kwargs: Any) -> Chain:
    """Load chain from file."""
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    if file_path.suffix == '.json':
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError('File type must be json or yaml')
    if 'verbose' in kwargs:
        config['verbose'] = kwargs.pop('verbose')
    if 'memory' in kwargs:
        config['memory'] = kwargs.pop('memory')
    return load_chain_from_config(config, **kwargs)