from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def vrf_error_check(module, commands, responses):
    """Checks for VRF config errors and executes a retry in some circumstances."""
    pattern = 'ERROR: Deletion of VRF .* in progress'
    if re.search(pattern, str(responses)):
        time.sleep(15)
        responses = load_config(module, commands, opts={'catch_clierror': True})
        if re.search(pattern, str(responses)):
            module.fail_json(msg='VRF config (and retry) failure: %s ' % responses)
        module.warn('VRF config delayed by VRF deletion - passed on retry')