from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
def _verify_ed25519(self, msg_or_hash, signature, ph):
    if len(signature) != 64:
        raise ValueError('The signature is not authentic (length)')
    if self._context or ph:
        flag = int(ph)
        dom2 = b'SigEd25519 no Ed25519 collisions' + bchr(flag) + bchr(len(self._context)) + self._context
    else:
        dom2 = b''
    PHM = msg_or_hash.digest() if ph else msg_or_hash
    try:
        R = import_public_key(signature[:32]).pointQ
    except ValueError:
        raise ValueError('The signature is not authentic (R)')
    s = Integer.from_bytes(signature[32:], 'little')
    if s > self._order:
        raise ValueError('The signature is not authentic (S)')
    k_hash = SHA512.new(dom2 + signature[:32] + self._A + PHM).digest()
    k = Integer.from_bytes(k_hash, 'little') % self._order
    point1 = s * 8 * self._key._curve.G
    point2 = 8 * R + k * 8 * self._key.pointQ
    if point1 != point2:
        raise ValueError('The signature is not authentic')