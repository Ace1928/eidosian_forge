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
def _gpg_key_exists(self, key_fingerprint):
    found = False
    keyfiles = ['/etc/apt/trusted.gpg']
    for other_dir in APT_KEY_DIRS:
        keyfiles.extend([os.path.join(other_dir, x) for x in os.listdir(other_dir) if not x.startswith('.')])
    for key_file in keyfiles:
        if os.path.exists(key_file):
            try:
                rc, out, err = self.module.run_command([self.gpg_bin, '--list-packets', key_file])
            except (IOError, OSError) as e:
                self.debug('Could check key against file %s: %s' % (key_file, to_native(e)))
                continue
            if key_fingerprint in out:
                found = True
                break
    return found