from __future__ import annotations
import json
from json import JSONDecodeError
from time import sleep
from typing import (
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
def _is_assistants_builtin_tool(tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]) -> bool:
    """Determine if tool corresponds to OpenAI Assistants built-in."""
    assistants_builtin_tools = ('code_interpreter', 'retrieval')
    return isinstance(tool, dict) and 'type' in tool and (tool['type'] in assistants_builtin_tools)