from __future__ import annotations
import enum
import typing
import datetime
import croniter
from aiokeydb.v2.types.base import BaseModel, lazyproperty, Field, validator
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils.logs import logger
from aiokeydb.v2.types.static import JobStatus, TaskType, TERMINAL_STATUSES, UNSUCCESSFUL_TERMINAL_STATUSES, INCOMPLETE_STATUSES
def to_enqueue_kwargs(self, job_key: typing.Optional[str]=None, exclude_none: typing.Optional[bool]=True, **kwargs) -> typing.Dict[str, typing.Any]:
    """
        Returns the kwargs for the job
        """
    default_kwargs = self.default_kwargs or {}
    if kwargs:
        default_kwargs.update(kwargs)
    default_kwargs['key'] = job_key
    enqueue_kwargs = {'job_or_func': self.function_name, **default_kwargs}
    if self.callback:
        enqueue_kwargs['job_callback'] = self.callback
        enqueue_kwargs['job_callback_kwargs'] = self.callback_kwargs
    if self.bypass_lock is not None:
        enqueue_kwargs['bypass_lock'] = self.bypass_lock
    if exclude_none:
        enqueue_kwargs = {k: v for k, v in enqueue_kwargs.items() if v is not None}
    enqueue_kwargs['scheduled'] = self.next_scheduled()
    return enqueue_kwargs