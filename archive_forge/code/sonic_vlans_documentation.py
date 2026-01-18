from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vlans.vlans import VlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.vlans.vlans import Vlans

    Main entry point for module execution

    :returns: the result form module invocation
    