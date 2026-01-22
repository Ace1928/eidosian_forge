from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
def _compute_nonce(self, msg_hash):
    return Integer.random_range(min_inclusive=1, max_exclusive=self._key._curve.order, randfunc=self._randfunc)