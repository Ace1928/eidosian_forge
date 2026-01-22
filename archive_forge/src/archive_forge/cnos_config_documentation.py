from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import load_config, get_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
main entry point for module execution
    