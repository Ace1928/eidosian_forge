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
@validator('url', pre=True, allow_reuse=True)
@classmethod
def _validate_url(cls, val: Any) -> AnyHttpUrl:
    """Do some initial conversations for the URL before checking."""
    if isinstance(val, str):
        return cast(AnyHttpUrl, parse_obj_as(AnyHttpUrl, val))
    return cast(AnyHttpUrl, val)