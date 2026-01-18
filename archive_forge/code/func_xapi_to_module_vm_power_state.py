from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def xapi_to_module_vm_power_state(power_state):
    """Maps XAPI VM power states to module VM power states."""
    module_power_state_map = {'running': 'poweredon', 'halted': 'poweredoff', 'suspended': 'suspended', 'paused': 'paused'}
    return module_power_state_map.get(power_state)