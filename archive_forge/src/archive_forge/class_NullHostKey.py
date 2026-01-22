import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
class NullHostKey:
    """
    This class represents the Null Host Key for GSS-API Key Exchange as defined
    in `RFC 4462 Section 5
    <https://tools.ietf.org/html/rfc4462.html#section-5>`_
    """

    def __init__(self):
        self.key = ''

    def __str__(self):
        return self.key

    def get_name(self):
        return self.key