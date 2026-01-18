from __future__ import (absolute_import, division, print_function)
import os
import pty
import subprocess
import fcntl
import ansible.constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.compat import selectors
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
 terminate the connection; nothing to do here 