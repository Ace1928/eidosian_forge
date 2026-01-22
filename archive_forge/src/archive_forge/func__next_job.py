import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
def _next_job(self):
    if self.only_unclaimed:
        allowed_states = self._UNCLAIMED_JOB_STATES
    else:
        allowed_states = self._JOB_STATES
    job = None
    while self._jobs and job is None:
        maybe_job = self._jobs.popleft()
        try:
            if maybe_job.state in allowed_states:
                job = maybe_job
        except excp.JobFailure:
            self._logger.warn("Failed determining the state of job '%s'", maybe_job, exc_info=True)
        except excp.NotFound:
            if self._board_removal_func is not None:
                self._board_removal_func(maybe_job)
    return job