from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
def _from_private_key_file(self, filename, password):
    data = self._read_private_key_file('RSA', filename, password)
    self._decode_key(data)