import collections
import contextlib
import functools
import sys
import threading
import fasteners
import futurist
from kazoo import exceptions as k_exceptions
from kazoo.protocol import paths as k_paths
from kazoo.protocol import states as k_states
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow.conductors import base as c_base
from taskflow import exceptions as excp
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
def _remove_job(self, path):
    if path not in self._known_jobs:
        return False
    with self._job_cond:
        job = self._known_jobs.pop(path, None)
    if job is not None:
        LOG.debug("Removed job that was at path '%s'", path)
        self._try_emit(base.REMOVAL, details={'job': job})
        return True
    else:
        return False