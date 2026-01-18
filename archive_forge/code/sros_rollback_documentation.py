from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.community.network.plugins.module_utils.network.sros.sros import load_config, get_config, sros_argument_spec, check_args
 main entry point for module execution
    