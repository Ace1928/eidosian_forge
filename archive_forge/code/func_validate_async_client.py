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
@root_validator()
def validate_async_client(cls, values: dict) -> dict:
    if values['async_client'] is None:
        import openai
        api_key = values['client'].api_key
        values['async_client'] = openai.AsyncOpenAI(api_key=api_key)
    return values