import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Verifies the ``passwd`` against a given password hash.

    Returns True on success, raises InvalidkeyError on failure
    :param passwd_hash: saved password hash
    :type passwd_hash: bytes
    :param passwd: password to be checked
    :type passwd: bytes
    :return: success
    :rtype: boolean
    