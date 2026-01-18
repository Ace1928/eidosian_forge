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
def test_save_retry_results(self):
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopRetry('my retry'))
    s.save('my retry', 'a')
    s.save('my retry', 'b')
    history = s.get_retry_history('my retry')
    self.assertEqual([('a', {}), ('b', {})], list(history))
    self.assertEqual(['a', 'b'], list(history.provided_iter()))