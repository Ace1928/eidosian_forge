from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def validate_level(value, module):
    if value not in ('admin', 'operator'):
        module.fail_json(msg='level must be either admin or operator, got %s' % value)