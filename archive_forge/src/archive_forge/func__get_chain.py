from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
from langchain_core.pydantic_v1 import Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.powerbi.prompt import (
from langchain_community.tools.powerbi.tool import (
from langchain_community.utilities.powerbi import PowerBIDataset
def _get_chain(self) -> LLMChain:
    """Construct the chain based on the callback manager and model type."""
    from langchain.chains.llm import LLMChain
    if isinstance(self.llm, BaseLanguageModel):
        return LLMChain(llm=self.llm, callback_manager=self.callback_manager if self.callback_manager else None, prompt=PromptTemplate(template=SINGLE_QUESTION_TO_QUERY, input_variables=['tool_input', 'tables', 'schemas', 'examples']))
    system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(template=QUESTION_TO_QUERY_BASE, input_variables=['tables', 'schemas', 'examples']))
    human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(template=USER_INPUT, input_variables=['tool_input']))
    return LLMChain(llm=self.llm, callback_manager=self.callback_manager if self.callback_manager else None, prompt=ChatPromptTemplate.from_messages([system_prompt, human_prompt]))