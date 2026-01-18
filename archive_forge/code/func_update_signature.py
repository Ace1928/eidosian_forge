from __future__ import unicode_literals
import binascii
from pymacaroons import Caveat
from pymacaroons.utils import (
from .base_first_party import (
def update_signature(self, signature, caveat):
    return binascii.unhexlify(sign_first_party_caveat(signature, caveat.caveat_id_bytes))