import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
import nacl.signing
from paramiko.message import Message
from paramiko.pkey import PKey, OPENSSH_AUTH_MAGIC, _unpad_openssh
from paramiko.util import b
from paramiko.ssh_exception import SSHException, PasswordRequiredException
def verify_ssh_sig(self, data, msg):
    if msg.get_text() != self.name:
        return False
    try:
        self._verifying_key.verify(data, msg.get_binary())
    except nacl.exceptions.BadSignatureError:
        return False
    else:
        return True