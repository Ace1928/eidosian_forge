from __future__ import annotations
import asyncio
import logging
import time
from typing import (
from langchain_core.agents import (
from langchain_core.callbacks import (
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langchain.schema import RUN_KEY
from langchain.utilities.asyncio import asyncio_timeout
def update_iterations(self) -> None:
    """
        Increment the number of iterations and update the time elapsed.
        """
    self.iterations += 1
    self.time_elapsed = time.time() - self.start_time
    logger.debug(f'Agent Iterations: {self.iterations} ({self.time_elapsed:.2f}s elapsed)')