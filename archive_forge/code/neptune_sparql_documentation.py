from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_community.graphs import NeptuneRdfGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import SPARQL_QA_PROMPT
from langchain.chains.llm import LLMChain

        Generate SPARQL query, use it to retrieve a response from the gdb and answer
        the question.
        