import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _listeners_from_job(self, job, engine):
    listeners = super(ExecutorConductor, self)._listeners_from_job(job, engine)
    listeners.append(logging_listener.LoggingListener(engine, log=self._log))
    return listeners