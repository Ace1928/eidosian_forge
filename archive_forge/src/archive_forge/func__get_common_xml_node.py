from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def _get_common_xml_node(self, name):
    root_node = build_root_xml_node('interface')
    build_child_xml_node(root_node, 'name', name)
    intf_unit_node = build_child_xml_node(root_node, 'unit')
    return (root_node, intf_unit_node)