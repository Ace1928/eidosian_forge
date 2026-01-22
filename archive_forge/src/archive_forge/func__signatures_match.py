import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def _signatures_match(self, macaroon_signature, computed_signature):
    return constant_time_compare(convert_to_bytes(macaroon_signature), convert_to_bytes(computed_signature))