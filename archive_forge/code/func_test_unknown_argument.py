import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_unknown_argument(self):
    exc = self.assertRaises(TypeError, failure.Failure, exception_str='Woot!', traceback_str=None, exc_type_names=['Exception'], hi='hi there')
    expected = 'Failure.__init__ got unexpected keyword argument(s): hi'
    self.assertEqual(expected, str(exc))