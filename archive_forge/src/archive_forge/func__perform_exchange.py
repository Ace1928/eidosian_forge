import binascii
import hashlib
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from paramiko.message import Message
from paramiko.common import byte_chr
from paramiko.ssh_exception import SSHException
def _perform_exchange(self, peer_key):
    secret = self.key.exchange(peer_key)
    if constant_time.bytes_eq(secret, b'\x00' * 32):
        raise SSHException("peer's curve25519 public value has wrong order")
    return secret