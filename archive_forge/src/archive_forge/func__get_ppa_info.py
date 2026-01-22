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
def _get_ppa_info(self, owner_name, ppa_name):
    lp_api = self.LP_API % (owner_name, ppa_name)
    headers = dict(Accept='application/json')
    response, info = fetch_url(self.module, lp_api, headers=headers)
    if info['status'] != 200:
        self.module.fail_json(msg='failed to fetch PPA information, error was: %s' % info['msg'])
    return json.loads(to_native(response.read()))