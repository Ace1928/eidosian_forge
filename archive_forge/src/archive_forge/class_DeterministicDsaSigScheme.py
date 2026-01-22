from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
class DeterministicDsaSigScheme(DssSigScheme):

    def __init__(self, key, encoding, order, private_key):
        super(DeterministicDsaSigScheme, self).__init__(key, encoding, order)
        self._private_key = private_key

    def _bits2int(self, bstr):
        """See 2.3.2 in RFC6979"""
        result = Integer.from_bytes(bstr)
        q_len = self._order.size_in_bits()
        b_len = len(bstr) * 8
        if b_len > q_len:
            result >>= b_len - q_len
        return result

    def _int2octets(self, int_mod_q):
        """See 2.3.3 in RFC6979"""
        assert 0 < int_mod_q < self._order
        return long_to_bytes(int_mod_q, self._order_bytes)

    def _bits2octets(self, bstr):
        """See 2.3.4 in RFC6979"""
        z1 = self._bits2int(bstr)
        if z1 < self._order:
            z2 = z1
        else:
            z2 = z1 - self._order
        return self._int2octets(z2)

    def _compute_nonce(self, mhash):
        """Generate k in a deterministic way"""
        h1 = mhash.digest()
        mask_v = b'\x01' * mhash.digest_size
        nonce_k = b'\x00' * mhash.digest_size
        for int_oct in (b'\x00', b'\x01'):
            nonce_k = HMAC.new(nonce_k, mask_v + int_oct + self._int2octets(self._private_key) + self._bits2octets(h1), mhash).digest()
            mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()
        nonce = -1
        while not 0 < nonce < self._order:
            if nonce != -1:
                nonce_k = HMAC.new(nonce_k, mask_v + b'\x00', mhash).digest()
                mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()
            mask_t = b''
            while len(mask_t) < self._order_bytes:
                mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()
                mask_t += mask_v
            nonce = self._bits2int(mask_t)
        return nonce

    def _valid_hash(self, msg_hash):
        return True