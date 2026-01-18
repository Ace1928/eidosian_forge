import hashlib
import hmac
from oslotest import base as test_base
import testscenarios
from oslo_utils import secretutils
def test_none_data_raises_type_error(self):
    self.assertRaises(TypeError, hashlib.md5, None)
    self.assertRaises(TypeError, secretutils.md5, None)
    self.assertRaises(TypeError, secretutils.md5, None, usedforsecurity=True)
    self.assertRaises(TypeError, secretutils.md5, None, usedforsecurity=False)