from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
@classmethod
def _gen_key_initctr(cls, b_password, b_salt):
    key_length = 32
    if HAS_CRYPTOGRAPHY:
        iv_length = algorithms.AES.block_size // 8
        b_derivedkey = cls._create_key_cryptography(b_password, b_salt, key_length, iv_length)
        b_iv = b_derivedkey[key_length * 2:key_length * 2 + iv_length]
    else:
        raise AnsibleError(NEED_CRYPTO_LIBRARY + '(Detected in initctr)')
    b_key1 = b_derivedkey[:key_length]
    b_key2 = b_derivedkey[key_length:key_length * 2]
    return (b_key1, b_key2, b_iv)