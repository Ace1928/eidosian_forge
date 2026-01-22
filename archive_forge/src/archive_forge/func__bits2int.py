from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
def _bits2int(self, bstr):
    """See 2.3.2 in RFC6979"""
    result = Integer.from_bytes(bstr)
    q_len = self._order.size_in_bits()
    b_len = len(bstr) * 8
    if b_len > q_len:
        result >>= b_len - q_len
    return result