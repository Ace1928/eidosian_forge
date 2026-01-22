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
def ask_vault_passwords(self):
    b_vault_passwords = []
    for prompt_format in self.prompt_formats:
        prompt = prompt_format % {'vault_id': self.vault_id}
        try:
            vault_pass = display.prompt(prompt, private=True)
        except EOFError:
            raise AnsibleVaultError('EOFError (ctrl-d) on prompt for (%s)' % self.vault_id)
        verify_secret_is_not_empty(vault_pass)
        b_vault_pass = to_bytes(vault_pass, errors='strict', nonstring='simplerepr').strip()
        b_vault_passwords.append(b_vault_pass)
    for b_vault_password in b_vault_passwords:
        self.confirm(b_vault_passwords[0], b_vault_password)
    if b_vault_passwords:
        return b_vault_passwords[0]
    return None