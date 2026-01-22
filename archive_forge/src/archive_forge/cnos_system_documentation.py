from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_config, load_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args, debugOutput
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList
 main entry point for module execution
    