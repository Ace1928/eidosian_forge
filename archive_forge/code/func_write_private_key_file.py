from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
def write_private_key_file(self, filename, password=None):
    self._write_private_key_file(filename, self.key, serialization.PrivateFormat.TraditionalOpenSSL, password=password)