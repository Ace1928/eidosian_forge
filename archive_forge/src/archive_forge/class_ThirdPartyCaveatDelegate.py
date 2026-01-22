from __future__ import unicode_literals
import binascii
from nacl.secret import SecretBox
from pymacaroons import Caveat
from pymacaroons.utils import (
from pymacaroons.exceptions import MacaroonUnmetCaveatException
from .base_third_party import (
class ThirdPartyCaveatDelegate(BaseThirdPartyCaveatDelegate):

    def __init__(self, *args, **kwargs):
        super(ThirdPartyCaveatDelegate, self).__init__(*args, **kwargs)

    def add_third_party_caveat(self, macaroon, location, key, key_id, **kwargs):
        derived_key = truncate_or_pad(generate_derived_key(convert_to_bytes(key)))
        old_key = truncate_or_pad(binascii.unhexlify(macaroon.signature_bytes))
        box = SecretBox(key=old_key)
        verification_key_id = box.encrypt(derived_key, nonce=kwargs.get('nonce'))
        caveat = Caveat(caveat_id=key_id, location=location, verification_key_id=verification_key_id, version=macaroon.version)
        macaroon.caveats.append(caveat)
        encode_key = binascii.unhexlify(macaroon.signature_bytes)
        macaroon.signature = sign_third_party_caveat(encode_key, caveat._verification_key_id, caveat._caveat_id)
        return macaroon