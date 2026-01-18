import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_reraises(self):
    exc = self.assertRaises(exceptions.WrappedFailure, self.fail_obj.reraise)
    self.assertIs(exc.check(RuntimeError), RuntimeError)