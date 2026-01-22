from __future__ import absolute_import, division, print_function
import collections
import json
from contextlib import contextmanager
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
def _validate_rollback_id(module, value):
    try:
        if not 0 <= int(value) <= 49:
            raise ValueError
    except ValueError:
        module.fail_json(msg='rollback must be between 0 and 49')