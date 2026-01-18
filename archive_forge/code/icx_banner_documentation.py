from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import exec_command
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError
entry point for module execution
    