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
def _decrypt_cryptography(cls, b_ciphertext, b_crypted_hmac, b_key1, b_key2, b_iv):
    hmac = HMAC(b_key2, hashes.SHA256(), CRYPTOGRAPHY_BACKEND)
    hmac.update(b_ciphertext)
    try:
        hmac.verify(_unhexlify(b_crypted_hmac))
    except InvalidSignature as e:
        raise AnsibleVaultError('HMAC verification failed: %s' % e)
    cipher = C_Cipher(algorithms.AES(b_key1), modes.CTR(b_iv), CRYPTOGRAPHY_BACKEND)
    decryptor = cipher.decryptor()
    unpadder = padding.PKCS7(128).unpadder()
    b_plaintext = unpadder.update(decryptor.update(b_ciphertext) + decryptor.finalize()) + unpadder.finalize()
    return b_plaintext