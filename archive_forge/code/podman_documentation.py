from __future__ import (absolute_import, division, print_function)
import os
import shlex
import shutil
import subprocess
from ansible.module_utils.common.process import get_bin_path
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.utils.display import Display
 unmount container's filesystem 