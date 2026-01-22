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
@property
def _is_primary_worker(self) -> bool:
    """
        Returns whether the worker is the primary worker.
        """
    return self.is_leader_process or self.name[-1] == '0'