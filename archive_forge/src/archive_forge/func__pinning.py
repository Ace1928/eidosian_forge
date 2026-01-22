from __future__ import absolute_import, division, print_function
import hashlib
import io
import json
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule, to_bytes
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.jenkins import download_updates_file
def _pinning(self, action):
    changed = False
    if action == 'pin' and (not self.is_pinned) or (action == 'unpin' and self.is_pinned):
        if not self.module.check_mode:
            self._pm_query(action, '%sning' % action.capitalize())
        changed = True
    return changed