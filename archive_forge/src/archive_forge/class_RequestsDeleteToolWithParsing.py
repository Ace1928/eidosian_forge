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
class RequestsDeleteToolWithParsing(BaseRequestsTool, BaseTool):
    """Tool that sends a DELETE request and parses the response."""
    name: str = 'requests_delete'
    'The name of the tool.'
    description = REQUESTS_DELETE_TOOL_DESCRIPTION
    'The description of the tool.'
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    'The maximum length of the response.'
    llm_chain: Any = Field(default_factory=_get_default_llm_chain_factory(PARSING_DELETE_PROMPT))
    'The LLM chain used to parse the response.'

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown
        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(str, self.requests_wrapper.delete(data['url']))
        response = response[:self.response_length]
        return self.llm_chain.predict(response=response, instructions=data['output_instructions']).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()