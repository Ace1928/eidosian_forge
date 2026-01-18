from __future__ import absolute_import, division, print_function
import base64
import hashlib
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def validate_privilege(value, module):
    if value and (not 1 <= value <= 15):
        module.fail_json(msg='privilege must be between 1 and 15, got %s' % value)