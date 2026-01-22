from __future__ import (annotations, absolute_import, division, print_function)
import os
import socket
import tempfile
import traceback
import fcntl
import re
import typing as t
from ansible.module_utils.compat.version import LooseVersion
from binascii import hexlify
from ansible.errors import (
from ansible.module_utils.compat.paramiko import PARAMIKO_IMPORT_ERR, paramiko
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def _save_ssh_host_keys(self, filename: str) -> None:
    """
        not using the paramiko save_ssh_host_keys function as we want to add new SSH keys at the bottom so folks
        don't complain about it :)
        """
    if not self._any_keys_added():
        return
    path = os.path.expanduser('~/.ssh')
    makedirs_safe(path)
    with open(filename, 'w') as f:
        for hostname, keys in self.ssh._host_keys.items():
            for keytype, key in keys.items():
                added_this_time = getattr(key, '_added_by_ansible_this_time', False)
                if not added_this_time:
                    f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))
        for hostname, keys in self.ssh._host_keys.items():
            for keytype, key in keys.items():
                added_this_time = getattr(key, '_added_by_ansible_this_time', False)
                if added_this_time:
                    f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))