from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
def _suggest_filename(self, line):

    def _cleanup_filename(s):
        filename = self.module.params['filename']
        if filename is not None:
            return filename
        return '_'.join(re.sub('[^a-zA-Z0-9]', ' ', s).split())

    def _strip_username_password(s):
        if '@' in s:
            s = s.split('@', 1)
            s = s[-1]
        return s
    line = re.sub('\\[[^\\]]+\\]', '', line)
    line = re.sub('\\w+://', '', line)
    parts = [part for part in line.split() if part not in VALID_SOURCE_TYPES]
    parts[0] = _strip_username_password(parts[0])
    return '%s.list' % _cleanup_filename(' '.join(parts[:1]))