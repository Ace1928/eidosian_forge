from base64 import standard_b64encode
from pymacaroons.utils import convert_to_string, convert_to_bytes
@verification_key_id.setter
def verification_key_id(self, value):
    self._verification_key_id = convert_to_bytes(value)