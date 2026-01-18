from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def unconfig_evpn_instance(self):
    """Unconfigure EVPN instance"""
    self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
    xml_str = CE_NC_MERGE_EVPN_CONFIG_HEAD % (self.bridge_domain_id, self.bridge_domain_id)
    self.updates_cmd.append('  evpn')
    if self.route_distinguisher:
        if self.route_distinguisher.lower() == 'auto':
            xml_str += '<evpnAutoRD>false</evpnAutoRD>'
            self.updates_cmd.append('    undo route-distinguisher auto')
        else:
            xml_str += '<evpnRD></evpnRD>'
            self.updates_cmd.append('    undo route-distinguisher %s' % self.route_distinguisher)
        xml_str += CE_NC_MERGE_EVPN_CONFIG_TAIL
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'UNDO_EVPN_BD_RD')
        self.changed = True
        return
    vpn_target_export = copy.deepcopy(self.vpn_target_export)
    vpn_target_import = copy.deepcopy(self.vpn_target_import)
    if self.vpn_target_both:
        for ele in self.vpn_target_both:
            if ele not in vpn_target_export:
                vpn_target_export.append(ele)
            if ele not in vpn_target_import:
                vpn_target_import.append(ele)
    head_flag = False
    if vpn_target_export:
        for ele in vpn_target_export:
            if ele.lower() == 'auto':
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                    head_flag = True
                xml_str += CE_NC_DELETE_EVPN_AUTORTS_CONTEXT % 'export_extcommunity'
                self.updates_cmd.append('    undo vpn-target auto export-extcommunity')
    if vpn_target_import:
        for ele in vpn_target_import:
            if ele.lower() == 'auto':
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                    head_flag = True
                xml_str += CE_NC_DELETE_EVPN_AUTORTS_CONTEXT % 'import_extcommunity'
                self.updates_cmd.append('    undo vpn-target auto import-extcommunity')
    if head_flag:
        xml_str += CE_NC_MERGE_EVPN_AUTORTS_TAIL
    head_flag = False
    if vpn_target_export:
        for ele in vpn_target_export:
            if ele.lower() != 'auto':
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                    head_flag = True
                xml_str += CE_NC_DELETE_EVPN_RTS_CONTEXT % ('export_extcommunity', ele)
                self.updates_cmd.append('    undo vpn-target %s export-extcommunity' % ele)
    if vpn_target_import:
        for ele in vpn_target_import:
            if ele.lower() != 'auto':
                if not head_flag:
                    xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                    head_flag = True
                xml_str += CE_NC_DELETE_EVPN_RTS_CONTEXT % ('import_extcommunity', ele)
                self.updates_cmd.append('    undo vpn-target %s import-extcommunity' % ele)
    if head_flag:
        xml_str += CE_NC_MERGE_EVPN_RTS_TAIL
    xml_str += CE_NC_MERGE_EVPN_CONFIG_TAIL
    recv_xml = set_nc_config(self.module, xml_str)
    self.check_response(recv_xml, 'MERGE_EVPN_BD_VPN_TARGET_CONFIG')
    self.changed = True