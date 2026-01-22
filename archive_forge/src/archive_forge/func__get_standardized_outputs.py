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
def _get_standardized_outputs(run: Run, schema_format: Literal['original', 'streaming_events']) -> Optional[Any]:
    """Extract standardized output from a run.

    Standardizes the outputs based on the type of the runnable used.

    Args:
        log: The log entry.
        schema_format: The schema format to use.

    Returns:
        An output if returned, otherwise a None
    """
    outputs = load(run.outputs)
    if schema_format == 'original':
        if run.run_type == 'prompt' and 'output' in outputs:
            return outputs['output']
        return outputs
    if run.run_type in {'retriever', 'llm', 'chat_model'}:
        return outputs
    if isinstance(outputs, dict):
        return outputs.get('output', None)
    return None