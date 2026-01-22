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
def _input_iterator() -> Iterator[TTSInputType]:
    """Iterate over the input_queue."""
    while True:
        try:
            val = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if val == _TRANSFORM_END:
            break
        yield val