from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import run_commands, get_config, load_config, get_diff
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
 main entry point for module execution
    