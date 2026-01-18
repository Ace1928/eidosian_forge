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
def trim_query(query: str) -> str:
    """Trim the query to only include Cypher keywords."""
    keywords = ('CALL', 'CREATE', 'DELETE', 'DETACH', 'LIMIT', 'MATCH', 'MERGE', 'OPTIONAL', 'ORDER', 'REMOVE', 'RETURN', 'SET', 'SKIP', 'UNWIND', 'WITH', 'WHERE', '//')
    lines = query.split('\n')
    new_query = ''
    for line in lines:
        if line.strip().upper().startswith(keywords):
            new_query += line + '\n'
    return new_query