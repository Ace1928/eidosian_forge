import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_reraises_several(self):
    fls = [_captured_failure('Woot!'), _captured_failure('Oh, not again!')]
    exc = self.assertRaises(exceptions.WrappedFailure, failure.Failure.reraise_if_any, fls)
    self.assertEqual(fls, list(exc))