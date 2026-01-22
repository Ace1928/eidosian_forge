import binascii
import json
from pymacaroons import utils
def _read_json_binary_field(deserialized, field):
    """ Read the value of a JSON field that may be string or base64-encoded.
    """
    val = deserialized.get(field)
    if val is not None:
        return utils.convert_to_bytes(val)
    val = deserialized.get(field + '64')
    if val is None:
        return None
    return utils.raw_urlsafe_b64decode(val)