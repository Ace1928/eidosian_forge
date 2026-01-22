from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible.module_utils._text import to_text
import re
 main entry point for module execution
    