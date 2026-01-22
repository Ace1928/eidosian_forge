import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast
import yaml
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool, Tool
from langchain_community.agent_toolkits.openapi.planner_prompt import (
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.llms import OpenAI
from langchain_community.tools.requests.tool import BaseRequestsTool
from langchain_community.utilities.requests import RequestsWrapper
def _get_default_llm_chain_factory(prompt: BasePromptTemplate) -> Callable[[], Any]:
    """Returns a default LLMChain factory."""
    return partial(_get_default_llm_chain, prompt)