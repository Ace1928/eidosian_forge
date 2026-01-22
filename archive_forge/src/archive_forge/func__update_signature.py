import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def _update_signature(self, caveat, signature):
    if caveat.first_party():
        return self.first_party_caveat_verifier_delegate.update_signature(signature, caveat)
    else:
        return self.third_party_caveat_verifier_delegate.update_signature(signature, caveat)