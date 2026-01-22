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
class RequestsPatchToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests PATCH tool with LLM-instructed extraction of truncated responses."""
    name: str = 'requests_patch'
    'Tool name.'
    description = REQUESTS_PATCH_TOOL_DESCRIPTION
    'Tool description.'
    response_length: int = MAX_RESPONSE_LENGTH
    'Maximum length of the response to be returned.'
    llm_chain: Any = Field(default_factory=_get_default_llm_chain_factory(PARSING_PATCH_PROMPT))
    'LLMChain used to extract the response.'

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown
        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(str, self.requests_wrapper.patch(data['url'], data['data']))
        response = response[:self.response_length]
        return self.llm_chain.predict(response=response, instructions=data['output_instructions']).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()