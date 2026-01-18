from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.config.firewall_rules.firewall_rules import (

    Main entry point for module execution
    :returns: the result form module invocation
    