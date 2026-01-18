import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_valid_from_dict_to_dict_2(self):
    f = _captured_failure('Woot!')
    d_f = f.to_dict()
    d_f['exc_type_names'] = ['RuntimeError', 'Exception', 'BaseException']
    failure.Failure.validate(d_f)