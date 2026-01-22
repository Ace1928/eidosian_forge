from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config, ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
 The work function 