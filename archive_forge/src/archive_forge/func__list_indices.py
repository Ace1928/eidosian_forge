from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.elasticsearch_database.prompts import ANSWER_PROMPT, DSL_PROMPT
from langchain.chains.llm import LLMChain
def _list_indices(self) -> List[str]:
    all_indices = [index['index'] for index in self.database.cat.indices(format='json')]
    if self.include_indices:
        all_indices = [i for i in all_indices if i in self.include_indices]
    if self.ignore_indices:
        all_indices = [i for i in all_indices if i not in self.ignore_indices]
    return all_indices