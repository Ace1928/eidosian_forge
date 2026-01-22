from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def _patch_host_update(self):
    _host_methods = self.foremanapi.apidoc['docs']['resources']['hosts']['methods']
    _host_update = next((x for x in _host_methods if x['name'] == 'update'))
    for param in ['location_id', 'organization_id']:
        _host_update_taxonomy_param = next((x for x in _host_update['params'] if x['name'] == param), None)
        if _host_update_taxonomy_param is not None:
            _host_update['params'].remove(_host_update_taxonomy_param)