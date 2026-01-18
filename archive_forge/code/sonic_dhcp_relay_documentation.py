from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.dhcp_relay.dhcp_relay import Dhcp_relayArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.dhcp_relay.dhcp_relay import Dhcp_relay

    Main entry point for module execution

    :returns: the result form module invocation
    