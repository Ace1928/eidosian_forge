import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
@mock.patch('osprofiler._utils.generate_hmac')
def test_singed_unpack_generate_hmac_failed(self, mock_generate_hmac):
    mock_generate_hmac.side_effect = Exception
    self.assertIsNone(utils.signed_unpack('data', 'hmac_data', 'hmac_key'))