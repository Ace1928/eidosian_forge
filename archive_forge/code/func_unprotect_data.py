import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
def unprotect_data(keys, signed_data):
    """De-serialize data given a dict of keys.

    Given keys and cached string data, verifies the signature, decrypts if
    necessary, and returns the original serialized data.

    """
    if signed_data is None:
        return None
    provided_mac = signed_data[:DIGEST_LENGTH_B64]
    calculated_mac = sign_data(keys['MAC'], signed_data[DIGEST_LENGTH_B64:])
    if not secretutils.constant_time_compare(provided_mac, calculated_mac):
        raise InvalidMacError(_('Invalid MAC; data appears to be corrupted.'))
    data = base64.b64decode(signed_data[DIGEST_LENGTH_B64:])
    if keys['strategy'] == b'ENCRYPT':
        data = decrypt_data(keys['ENCRYPTION'], data)
    return data