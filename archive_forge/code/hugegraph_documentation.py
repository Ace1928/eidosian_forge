from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_community.graphs.hugegraph import HugeGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain
Generate gremlin statement, use it to look up in db and answer question.