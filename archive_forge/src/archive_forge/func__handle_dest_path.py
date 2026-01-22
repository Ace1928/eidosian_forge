from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
import uuid
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _handle_dest_path(self, dest):
    working_path = self._get_working_path()
    if os.path.isabs(dest) or urlsplit('dest').scheme:
        dst = dest
    else:
        dst = self._loader.path_dwim_relative(working_path, '', dest)
    return dst