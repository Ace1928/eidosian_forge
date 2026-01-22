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
def _create_api_planner_tool(api_spec: ReducedOpenAPISpec, llm: BaseLanguageModel) -> Tool:
    from langchain.chains.llm import LLMChain
    endpoint_descriptions = [f'{name} {description}' for name, description, _ in api_spec.endpoints]
    prompt = PromptTemplate(template=API_PLANNER_PROMPT, input_variables=['query'], partial_variables={'endpoints': '- ' + '- '.join(endpoint_descriptions)})
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(name=API_PLANNER_TOOL_NAME, description=API_PLANNER_TOOL_DESCRIPTION, func=chain.run)
    return tool