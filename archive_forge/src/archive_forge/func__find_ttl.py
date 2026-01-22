from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_ttl(clc, module):
    """
        Validate that TTL is > 3600 if set, and fail if not
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: validated ttl
        """
    ttl = module.params.get('ttl')
    if ttl:
        if ttl <= 3600:
            return module.fail_json(msg=str('Ttl cannot be <= 3600'))
        else:
            ttl = clc.v2.time_utils.SecondsToZuluTS(int(time.time()) + ttl)
    return ttl