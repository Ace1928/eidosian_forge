from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from re import findall
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.validation import check_required_together
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import validate_ip_address
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_config, load_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
 main entry point for module execution
    