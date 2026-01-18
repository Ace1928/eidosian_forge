import contextlib
import threading
from oslo_utils import uuidutils
from taskflow import exceptions
from taskflow.persistence import backends
from taskflow.persistence import models
from taskflow import states
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import persistence_utils as p_utils
def test_task_metadata_update_with_none(self):
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.update_atom_metadata('my task', None)
    self.assertEqual(0.0, s.get_task_progress('my task'))
    s.set_task_progress('my task', 0.5)
    self.assertEqual(0.5, s.get_task_progress('my task'))
    s.update_atom_metadata('my task', None)
    self.assertEqual(0.5, s.get_task_progress('my task'))