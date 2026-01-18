from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_community.graphs import BaseNeptuneGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
Generate Cypher statement, use it to look up in db and answer question.