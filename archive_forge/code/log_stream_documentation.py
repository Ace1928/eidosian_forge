from __future__ import annotations
import asyncio
import copy
import threading
from collections import defaultdict
from typing import (
from uuid import UUID
import jsonpatch  # type: ignore[import]
from typing_extensions import NotRequired, TypedDict
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.tracers.schemas import Run
Process new LLM token.