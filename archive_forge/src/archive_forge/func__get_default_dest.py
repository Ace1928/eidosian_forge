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
def _get_default_dest(self, src_path):
    dest_path = self._get_working_path()
    src_fname = self._get_src_filename_from_path(src_path)
    filename = '%s/%s' % (dest_path, src_fname)
    return filename