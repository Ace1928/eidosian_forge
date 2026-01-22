from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from openstack import exceptions
from openstack.image.iterable_chunked_file import IterableChunkedFile
class ImageSigner:
    """Image file signature generator.

    Generates signatures for files using a specified private key file.
    """

    def __init__(self, hash_method='SHA-256', padding_method='RSA-PSS'):
        padding_types = {'RSA-PSS': padding.PSS(mgf=padding.MGF1(HASH_METHODS[hash_method]), salt_length=padding.PSS.MAX_LENGTH)}
        self.hash_method = hash_method
        self.padding_method = padding_method
        self.private_key = None
        self.hash = HASH_METHODS[hash_method]
        self.hasher = hashes.Hash(self.hash, default_backend())
        self.padding = padding_types[padding_method]

    def load_private_key(self, file_path, password=None):
        with open(file_path, 'rb') as key_file:
            self.private_key = serialization.load_pem_private_key(key_file.read(), password=password, backend=default_backend())

    def generate_signature(self, file_obj):
        if not self.private_key:
            raise exceptions.SDKException('private_key not set')
        file_obj.seek(0)
        chunked_file = IterableChunkedFile(file_obj)
        for chunk in chunked_file:
            self.hasher.update(chunk)
        file_obj.seek(0)
        digest = self.hasher.finalize()
        signature = self.private_key.sign(digest, self.padding, utils.Prehashed(self.hash))
        return signature