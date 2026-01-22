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
def _key_already_exists(self, key_fingerprint):
    if self.apt_key_bin:
        locale = get_best_parsable_locale(self.module)
        APT_ENV = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LC_CTYPE=locale)
        self.module.run_command_environ_update = APT_ENV
        rc, out, err = self.module.run_command([self.apt_key_bin, 'export', key_fingerprint], check_rc=True)
        found = bool(not err or 'nothing exported' not in err)
    else:
        found = self._gpg_key_exists(key_fingerprint)
    return found