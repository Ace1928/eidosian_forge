from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import opaque_data
from castellan.common.objects import passphrase
from castellan.common.objects import private_key
from castellan.common.objects import public_key
from castellan.common.objects import symmetric_key
from castellan.common.objects import x_509
from castellan.tests import utils
def _get_test_private_key():
    key_bytes = bytes(utils.get_private_key_der())
    bit_length = 2048
    key = private_key.PrivateKey('RSA', bit_length, key_bytes)
    return key