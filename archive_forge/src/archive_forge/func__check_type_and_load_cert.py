import base64
from base64 import encodebytes, decodebytes
from binascii import unhexlify
import os
from pathlib import Path
from hashlib import md5, sha256
import re
import struct
import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import algorithms, modes, Cipher
from cryptography.hazmat.primitives import asymmetric
from paramiko import util
from paramiko.util import u, b
from paramiko.common import o600
from paramiko.ssh_exception import SSHException, PasswordRequiredException
from paramiko.message import Message
def _check_type_and_load_cert(self, msg, key_type, cert_type):
    """
        Perform message type-checking & optional certificate loading.

        This includes fast-forwarding cert ``msg`` objects past the nonce, so
        that the subsequent fields are the key numbers; thus the caller may
        expect to treat the message as key material afterwards either way.

        The obtained key type is returned for classes which need to know what
        it was (e.g. ECDSA.)
        """
    key_types = key_type
    cert_types = cert_type
    if isinstance(key_type, str):
        key_types = [key_types]
    if isinstance(cert_types, str):
        cert_types = [cert_types]
    if msg is None:
        raise SSHException('Key object may not be empty')
    msg.rewind()
    type_ = msg.get_text()
    if type_ in key_types:
        pass
    elif type_ in cert_types:
        self.load_certificate(Message(msg.asbytes()))
        msg.get_string()
    else:
        err = 'Invalid key (class: {}, data type: {}'
        raise SSHException(err.format(self.__class__.__name__, type_))