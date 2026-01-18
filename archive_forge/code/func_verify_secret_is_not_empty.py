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
def verify_secret_is_not_empty(secret, msg=None):
    """Check the secret against minimal requirements.

    Raises: AnsibleVaultPasswordError if the password does not meet requirements.

    Currently, only requirement is that the password is not None or an empty string.
    """
    msg = msg or 'Invalid vault password was provided'
    if not secret:
        raise AnsibleVaultPasswordError(msg)