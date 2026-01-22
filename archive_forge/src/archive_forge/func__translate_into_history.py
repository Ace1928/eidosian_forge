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
def _translate_into_history(self, ad):
    failure = None
    if ad.failure is not None:
        failure = ad.failure
        fail_cache = self._failures[ad.name]
        try:
            fail = fail_cache[states.EXECUTE]
            if failure.matches(fail):
                failure = fail
        except KeyError:
            pass
    return retry.History(ad.results, failure=failure)