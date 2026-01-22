import binascii
import json
from pymacaroons import utils
def _caveat_v1_to_dict(c):
    """ Return a caveat as a dictionary for export as the JSON
    macaroon v1 format.
    """
    serialized = {}
    if len(c.caveat_id) > 0:
        serialized['cid'] = c.caveat_id
    if c.verification_key_id:
        serialized['vid'] = utils.raw_urlsafe_b64encode(c.verification_key_id).decode('utf-8')
    if c.location:
        serialized['cl'] = c.location
    return serialized