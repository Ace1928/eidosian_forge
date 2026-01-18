from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.ping.ping import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.config.ping.ping import Ping

    Main entry point for module execution

    :returns: the result form module invocation
    