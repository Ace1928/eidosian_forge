import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from redis import WatchError
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow
def _enqueue_job(self, job: 'Job', pipeline: Optional['Pipeline']=None, at_front: bool=False) -> Job:
    """Enqueues a job for delayed execution without checking dependencies.

        If Queue is instantiated with is_async=False, job is executed immediately.

        Args:
            job (Job): The job to enqueue
            pipeline (Optional[Pipeline], optional): The Redis pipeline to use. Defaults to None.
            at_front (bool, optional): Whether should enqueue at the front of the queue. Defaults to False.

        Returns:
            Job: The enqued job
        """
    pipe = pipeline if pipeline is not None else self.connection.pipeline()
    pipe.sadd(self.redis_queues_keys, self.key)
    job.redis_server_version = self.get_redis_server_version()
    job.set_status(JobStatus.QUEUED, pipeline=pipe)
    job.origin = self.name
    job.enqueued_at = utcnow()
    if job.timeout is None:
        job.timeout = self._default_timeout
    job.save(pipeline=pipe)
    job.cleanup(ttl=job.ttl, pipeline=pipe)
    if self._is_async:
        self.push_job_id(job.id, pipeline=pipe, at_front=at_front)
    if pipeline is None:
        pipe.execute()
    if not self._is_async:
        job = self.run_sync(job)
    return job