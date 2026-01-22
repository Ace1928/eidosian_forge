import os
import asyncio
import signal
import typing
import contextlib
import functools
from croniter import croniter
from aiokeydb.v2.exceptions import ConnectionError
from aiokeydb.v2.configs import settings as default_settings
from aiokeydb.v2.types.jobs import Job, CronJob, JobStatus, TaskType
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.utils.logs import logger, ColorMap
def _broadcast_process(self, previous_task=None):
    """
        This is a separate process that runs in the background to process broadcasts.
        """
    if previous_task and isinstance(previous_task, asyncio.Task):
        self.tasks.discard(previous_task)
    if not self.event.is_set():
        new_task = asyncio.create_task(self.process_broadcast())
        self.tasks.add(new_task)
        new_task.add_done_callback(self._broadcast_process)