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
def _get_execution_order(self, parent_run_id: Optional[str]=None) -> int:
    """Get the execution order for a run."""
    if parent_run_id is None:
        return 1
    parent_run = self.run_map.get(parent_run_id)
    if parent_run is None:
        logger.debug(f'Parent run with UUID {parent_run_id} not found.')
        return 1
    if parent_run.child_execution_order is None:
        raise TracerException(f'Parent run with UUID {parent_run_id} has no child execution order.')
    return parent_run.child_execution_order + 1