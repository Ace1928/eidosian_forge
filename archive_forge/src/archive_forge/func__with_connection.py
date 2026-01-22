import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
def _with_connection(self, functor, *args, **kwargs):
    with contextlib.closing(self._backend.get_connection()) as conn:
        return functor(conn, *args, **kwargs)