from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
def _bits2octets(self, bstr):
    """See 2.3.4 in RFC6979"""
    z1 = self._bits2int(bstr)
    if z1 < self._order:
        z2 = z1
    else:
        z2 = z1 - self._order
    return self._int2octets(z2)