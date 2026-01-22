import binascii
import json
from pymacaroons import utils
def _caveat_v2_to_dict(c):
    """ Return a caveat as a dictionary for export as the JSON
    macaroon v2 format.
    """
    serialized = {}
    if len(c.caveat_id_bytes) > 0:
        _add_json_binary_field(c.caveat_id_bytes, serialized, 'i')
    if c.verification_key_id:
        _add_json_binary_field(c.verification_key_id, serialized, 'v')
    if c.location:
        serialized['l'] = c.location
    return serialized