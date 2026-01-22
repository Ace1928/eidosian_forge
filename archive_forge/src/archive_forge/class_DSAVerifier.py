from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives import hashes
class DSAVerifier(object):

    def __init__(self, signature, hash_method, public_key):
        self._signature = signature
        self._hash_method = hash_method
        self._public_key = public_key
        self._hasher = hashes.Hash(hash_method, default_backend())

    def update(self, data):
        self._hasher.update(data)

    def verify(self):
        digest = self._hasher.finalize()
        self._public_key.verify(self._signature, digest, utils.Prehashed(self._hash_method))