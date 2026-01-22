import logging
import os
from taskflow import exceptions
from taskflow.listeners import base
from taskflow import states
def _claim_checker(self, state, details):
    if not self._has_been_lost():
        LOG.debug("Job '%s' is still claimed (actively owned by '%s')", self._job, self._owner)
    else:
        LOG.warning("Job '%s' has lost its claim (previously owned by '%s')", self._job, self._owner)
        self._on_job_loss(self._engine, state, details)