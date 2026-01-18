from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_retrieval_prompt import (
Default chain to use when router doesn't map input to one of the destinations.