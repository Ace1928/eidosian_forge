from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vlans.vlans import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.config.vlans.vlans import Vlans
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import get_connection

    Main entry point for module execution

    :returns: the result form module invocation
    