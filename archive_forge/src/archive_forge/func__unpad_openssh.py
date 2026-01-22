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
def _unpad_openssh(data):
    padding_length = data[-1]
    if 32 <= padding_length < 127:
        return data
    if padding_length > 15:
        raise SSHException('Invalid key')
    for i in range(padding_length):
        if data[i - padding_length] != i + 1:
            raise SSHException('Invalid key')
    return data[:-padding_length]