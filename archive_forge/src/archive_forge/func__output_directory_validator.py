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
@validator('output_directory')
@classmethod
def _output_directory_validator(cls, v: str) -> str:
    if v:
        dirpath = pathlib.Path(v)
        dirpath.mkdir(parents=True, exist_ok=True)
        return str(dirpath.absolute())
    return v