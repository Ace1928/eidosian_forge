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
@tenacity.retry(retry=tenacity.retry_if_exception_type(exception_types=exceptions.StorageFailure), stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS), wait=tenacity.wait_fixed(RETRY_WAIT_TIMEOUT), reraise=True)
def _save_atom_detail(self, conn, original_atom_detail, atom_detail):
    original_atom_detail.update(conn.update_atom_details(atom_detail))
    return original_atom_detail