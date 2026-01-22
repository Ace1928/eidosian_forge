from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
def _valid_hash(self, msg_hash):
    """Verify that the strength of the hash matches or exceeds
        the strength of the EC. We fail if the hash is too weak."""
    modulus_bits = self._key.pointQ.size_in_bits()
    sha224 = ('2.16.840.1.101.3.4.2.4', '2.16.840.1.101.3.4.2.7', '2.16.840.1.101.3.4.2.5')
    sha256 = ('2.16.840.1.101.3.4.2.1', '2.16.840.1.101.3.4.2.8', '2.16.840.1.101.3.4.2.6')
    sha384 = ('2.16.840.1.101.3.4.2.2', '2.16.840.1.101.3.4.2.9')
    sha512 = ('2.16.840.1.101.3.4.2.3', '2.16.840.1.101.3.4.2.10')
    shs = sha224 + sha256 + sha384 + sha512
    try:
        result = msg_hash.oid in shs
    except AttributeError:
        result = False
    return result