import saml2
from saml2 import SamlBase
class _DefaultSignature:

    def __init__(self, sign_alg=None, digest_alg=None):
        if sign_alg is None:
            self.sign_alg = sig_default
        else:
            self.sign_alg = sign_alg
        if digest_alg is None:
            self.digest_alg = digest_default
        else:
            self.digest_alg = digest_alg

    def __str__(self):
        return repr(self) + self.sign_alg