from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
Convenience constructor for instantiating from destination prompts.