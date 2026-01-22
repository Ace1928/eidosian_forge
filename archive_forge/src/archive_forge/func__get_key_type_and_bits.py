import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _get_key_type_and_bits(self, key):
    """
        Given any key, return its type/algorithm & bits-to-sign.

        Intended for input to or verification of, key signatures.
        """
    if key.public_blob:
        return (key.public_blob.key_type, key.public_blob.key_blob)
    else:
        return (key.get_name(), key)