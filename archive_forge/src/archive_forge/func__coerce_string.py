import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
def _coerce_string(val: 'TTSInputType') -> str:
    """Attempt to coerce the input value to a string.

    This is particularly useful for converting LangChain message to strings.
    """
    if isinstance(val, PromptValue):
        return val.to_string()
    if isinstance(val, BaseMessage):
        return str(val.content)
    return str(val)