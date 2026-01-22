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
def _on_job_posting(self, children, delayed=True):
    LOG.debug('Got children %s under path %s', children, self.path)
    child_paths = []
    for c in children:
        if c.endswith(self.LOCK_POSTFIX) or not c.startswith(self.JOB_PREFIX):
            continue
        child_paths.append(k_paths.join(self.path, c))
    investigate_paths = []
    pending_removals = []
    with self._job_cond:
        for path in self._known_jobs.keys():
            if path not in child_paths:
                pending_removals.append(path)
    for path in child_paths:
        if path in self._bad_paths:
            continue
        if path in self._known_jobs:
            continue
        if path not in investigate_paths:
            investigate_paths.append(path)
    if pending_removals:
        with self._job_cond:
            am_removed = 0
            try:
                for path in pending_removals:
                    am_removed += int(self._remove_job(path))
            finally:
                if am_removed:
                    self._job_cond.notify_all()
    for path in investigate_paths:
        request = self._client.get_async(path)
        if delayed:
            request.rawlink(functools.partial(self._process_child, path))
        else:
            self._process_child(path, request, quiet=False)