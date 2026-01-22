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
def _get_assistants_tool(tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Note that OpenAI assistants supports several built-in tools,
    such as "code_interpreter" and "retrieval."
    """
    if _is_assistants_builtin_tool(tool):
        return tool
    else:
        return convert_to_openai_tool(tool)