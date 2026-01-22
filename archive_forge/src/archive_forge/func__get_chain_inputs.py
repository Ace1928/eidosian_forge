from __future__ import annotations
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
from uuid import UUID
from tenacity import RetryCallState
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
from langchain_core.tracers.schemas import Run
def _get_chain_inputs(self, inputs: Any) -> Any:
    """Get the inputs for a chain run."""
    if self._schema_format == 'original':
        return inputs if isinstance(inputs, dict) else {'input': inputs}
    elif self._schema_format == 'streaming_events':
        return {'input': inputs}
    else:
        raise ValueError(f'Invalid format: {self._schema_format}')