from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_config, load_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import debugOutput, check_args
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
 main entry point for module execution
    