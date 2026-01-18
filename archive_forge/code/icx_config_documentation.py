from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import run_commands, get_config
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_defaults_flag, get_connection
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import check_args as icx_check_args
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
 main entry point for module execution
    