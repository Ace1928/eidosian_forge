from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import CustomNetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import (
 main entry point for module execution
    