from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
Check if the key ID is in pacman's keyring