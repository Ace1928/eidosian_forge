import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_signed_pack_unpack(self):
    hmac = 'secret'
    data = {'some': 'data'}
    packed_data, hmac_data = utils.signed_pack(data, hmac)
    process_data = utils.signed_unpack(packed_data, hmac_data, [hmac])
    self.assertIn('hmac_key', process_data)
    process_data.pop('hmac_key')
    self.assertEqual(data, process_data)