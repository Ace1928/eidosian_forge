from base64 import encodebytes, decodebytes
import binascii
import os
import re
from collections.abc import MutableMapping
from hashlib import sha1
from hmac import HMAC
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import get_logger, constant_time_bytes_eq, b, u
from paramiko.ssh_exception import SSHException
class InvalidHostKey(Exception):

    def __init__(self, line, exc):
        self.line = line
        self.exc = exc
        self.args = (line, exc)