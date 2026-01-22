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
@fasteners.read_locked
def _get_failures(self, fail_cache_key):
    failures = {}
    for atom_name, fail_cache in self._failures.items():
        try:
            failures[atom_name] = fail_cache[fail_cache_key]
        except KeyError:
            pass
    return failures