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
def test_save_fail_fetch_revert(self):
    t = test_utils.GiveBackRevert('my task')
    s = self._get_storage()
    s.ensure_atom(t)
    s.set_atom_intention('my task', states.REVERT)
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    s.save('my task', a_failure, state=states.REVERT_FAILURE)
    self.assertEqual(a_failure, s.get_revert_result('my task'))