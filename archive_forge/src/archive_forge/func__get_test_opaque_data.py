from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import opaque_data
from castellan.common.objects import passphrase
from castellan.common.objects import private_key
from castellan.common.objects import public_key
from castellan.common.objects import symmetric_key
from castellan.common.objects import x_509
from castellan.tests import utils
def _get_test_opaque_data():
    data = bytes(uuidutils.generate_uuid(), encoding='utf-8')
    opaque_object = opaque_data.OpaqueData(data)
    return opaque_object