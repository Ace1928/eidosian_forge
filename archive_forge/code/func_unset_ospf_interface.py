from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def unset_ospf_interface(self):
    """set interface ospf disable, and all its ospf attributes will be removed"""
    intf_dict = self.ospf_info['interface']
    xml_sum = ''
    xml_intf = CE_NC_XML_SET_IF_NAME % self.interface
    if intf_dict.get('silentEnable') == 'true':
        xml_sum += CE_NC_XML_BUILD_MERGE_INTF % (xml_intf + CE_NC_XML_SET_SILENT % 'false')
        self.updates_cmd.append('ospf %s' % self.process_id)
        self.updates_cmd.append('area %s' % self.get_area_ip())
        self.updates_cmd.append('undo silent-interface %s' % self.interface)
    xml_sum += CE_NC_XML_BUILD_DELETE_INTF % xml_intf
    xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, self.get_area_ip(), xml_sum)
    self.netconf_set_config(xml_str, 'DELETE_INTERFACE_OSPF')
    self.updates_cmd.append('undo ospf cost')
    self.updates_cmd.append('undo ospf timer hello')
    self.updates_cmd.append('undo ospf timer dead')
    self.updates_cmd.append('undo ospf authentication-mode')
    self.updates_cmd.append('undo ospf enable %s area %s' % (self.process_id, self.get_area_ip()))
    self.changed = True