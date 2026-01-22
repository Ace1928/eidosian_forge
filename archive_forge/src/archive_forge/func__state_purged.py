from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def _state_purged(self, want, have):
    """The command generator when state is deleted
        :rtype: A list
        :returns: the commands necessary to remove the current configuration
                  of the provided objects
        """
    bgp_xml = []
    delete = {'delete': 'delete'}
    build_child_xml_node(self.protocols, 'bgp', None, delete)
    autonomous_system = have.get('as_number')
    if autonomous_system:
        build_child_xml_node(self.routing_options, 'autonomous-system', None, {'delete': 'delete'})
    return bgp_xml