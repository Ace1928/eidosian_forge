import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def _caveat_met(self, root, caveat, macaroon, discharge_macaroons, signature):
    if caveat.first_party():
        return self.first_party_caveat_verifier_delegate.verify_first_party_caveat(self, caveat, signature)
    else:
        return self.third_party_caveat_verifier_delegate.verify_third_party_caveat(self, caveat, root, macaroon, discharge_macaroons, signature)