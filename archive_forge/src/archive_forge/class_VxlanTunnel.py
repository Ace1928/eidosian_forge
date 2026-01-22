from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class VxlanTunnel(object):
    """
    Manages vxlan tunnel configuration.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.bridge_domain_id = self.module.params['bridge_domain_id']
        self.vni_id = self.module.params['vni_id']
        self.nve_name = self.module.params['nve_name']
        self.nve_mode = self.module.params['nve_mode']
        self.peer_list_ip = self.module.params['peer_list_ip']
        self.protocol_type = self.module.params['protocol_type']
        self.source_ip = self.module.params['source_ip']
        self.state = self.module.params['state']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.existing = dict()
        self.proposed = dict()
        self.end_state = dict()
        self.vni2bd_info = None
        self.nve_info = None

    def init_module(self):
        """ init module """
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_vni2bd_dict(self):
        """ get vni2bd attributes dict."""
        vni2bd_info = dict()
        conf_str = CE_NC_GET_VNI_BD_INFO
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return vni2bd_info
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        vni2bd_info['vni2BdInfos'] = list()
        vni2bds = root.findall('nvo3/nvo3Vni2Bds/nvo3Vni2Bd')
        if vni2bds:
            for vni2bd in vni2bds:
                vni_dict = dict()
                for ele in vni2bd:
                    if ele.tag in ['vniId', 'bdId']:
                        vni_dict[ele.tag] = ele.text
                vni2bd_info['vni2BdInfos'].append(vni_dict)
        return vni2bd_info

    def check_nve_interface(self, nve_name):
        """is nve interface exist"""
        if not self.nve_info:
            return False
        if self.nve_info['ifName'] == nve_name:
            return True
        return False

    def get_nve_dict(self, nve_name):
        """ get nve interface attributes dict."""
        nve_info = dict()
        conf_str = CE_NC_GET_NVE_INFO % nve_name
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return nve_info
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        nvo3 = root.find('nvo3/nvo3Nves/nvo3Nve')
        if nvo3:
            for nve in nvo3:
                if nve.tag in ['srcAddr', 'ifName', 'nveType']:
                    nve_info[nve.tag] = nve.text
        nve_info['vni_peer_protocols'] = list()
        vni_members = root.findall('nvo3/nvo3Nves/nvo3Nve/vniMembers/vniMember')
        if vni_members:
            for member in vni_members:
                vni_dict = dict()
                for ele in member:
                    if ele.tag in ['vniId', 'protocol']:
                        vni_dict[ele.tag] = ele.text
                nve_info['vni_peer_protocols'].append(vni_dict)
        nve_info['vni_peer_ips'] = list()
        re_find = re.findall('<vniId>(.*?)</vniId>\\s*<protocol>(.*?)</protocol>\\s*<nvo3VniPeers>(.*?)</nvo3VniPeers>', xml_str)
        if re_find:
            for vni_peers in re_find:
                vni_info = dict()
                vni_peer = re.findall('<peerAddr>(.*?)</peerAddr>', vni_peers[2])
                if vni_peer:
                    vni_info['vniId'] = vni_peers[0]
                    vni_peer_list = list()
                    for peer in vni_peer:
                        vni_peer_list.append(peer)
                    vni_info['peerAddr'] = vni_peer_list
                nve_info['vni_peer_ips'].append(vni_info)
        return nve_info

    def check_nve_name(self):
        """Gets Nve interface name"""
        if self.nve_name is None:
            return False
        if self.nve_name in ['Nve1', 'Nve2']:
            return True
        return False

    def is_vni_bd_exist(self, vni_id, bd_id):
        """is vni to bridge-domain-id exist"""
        if not self.vni2bd_info:
            return False
        for vni2bd in self.vni2bd_info['vni2BdInfos']:
            if vni2bd['vniId'] == vni_id and vni2bd['bdId'] == bd_id:
                return True
        return False

    def is_vni_bd_change(self, vni_id, bd_id):
        """is vni to bridge-domain-id change"""
        if not self.vni2bd_info:
            return True
        for vni2bd in self.vni2bd_info['vni2BdInfos']:
            if vni2bd['vniId'] == vni_id and vni2bd['bdId'] == bd_id:
                return False
        return True

    def is_nve_mode_exist(self, nve_name, mode):
        """is nve interface mode exist"""
        if not self.nve_info:
            return False
        if self.nve_info['ifName'] == nve_name and self.nve_info['nveType'] == mode:
            return True
        return False

    def is_nve_mode_change(self, nve_name, mode):
        """is nve interface mode change"""
        if not self.nve_info:
            return True
        if self.nve_info['ifName'] == nve_name and self.nve_info['nveType'] == mode:
            return False
        return True

    def is_nve_source_ip_exist(self, nve_name, source_ip):
        """is vni to bridge-domain-id exist"""
        if not self.nve_info:
            return False
        if self.nve_info['ifName'] == nve_name and self.nve_info['srcAddr'] == source_ip:
            return True
        return False

    def is_nve_source_ip_change(self, nve_name, source_ip):
        """is vni to bridge-domain-id change"""
        if not self.nve_info:
            return True
        if self.nve_info['ifName'] == nve_name and self.nve_info['srcAddr'] == source_ip:
            return False
        return True

    def is_vni_protocol_exist(self, nve_name, vni_id, protocol_type):
        """is vni protocol exist"""
        if not self.nve_info:
            return False
        if self.nve_info['ifName'] == nve_name:
            for member in self.nve_info['vni_peer_protocols']:
                if member['vniId'] == vni_id and member['protocol'] == protocol_type:
                    return True
        return False

    def is_vni_protocol_change(self, nve_name, vni_id, protocol_type):
        """is vni protocol change"""
        if not self.nve_info:
            return True
        if self.nve_info['ifName'] == nve_name:
            for member in self.nve_info['vni_peer_protocols']:
                if member['vniId'] == vni_id and member['protocol'] == protocol_type:
                    return False
        return True

    def is_vni_peer_list_exist(self, nve_name, vni_id, peer_ip):
        """is vni peer list exist"""
        if not self.nve_info:
            return False
        if self.nve_info['ifName'] == nve_name:
            for member in self.nve_info['vni_peer_ips']:
                if member['vniId'] == vni_id and peer_ip in member['peerAddr']:
                    return True
        return False

    def is_vni_peer_list_change(self, nve_name, vni_id, peer_ip_list):
        """is vni peer list change"""
        if not self.nve_info:
            return True
        if self.nve_info['ifName'] == nve_name:
            if not self.nve_info['vni_peer_ips']:
                return True
            nve_peer_info = list()
            for nve_peer in self.nve_info['vni_peer_ips']:
                if nve_peer['vniId'] == vni_id:
                    nve_peer_info.append(nve_peer)
            if not nve_peer_info:
                return True
            nve_peer_list = nve_peer_info[0]['peerAddr']
            for peer in peer_ip_list:
                if peer not in nve_peer_list:
                    return True
            return False

    def config_merge_vni2bd(self, bd_id, vni_id):
        """config vni to bd id"""
        if self.is_vni_bd_change(vni_id, bd_id):
            cfg_xml = CE_NC_MERGE_VNI_BD_ID % (vni_id, bd_id)
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'MERGE_VNI_BD')
            self.updates_cmd.append('bridge-domain %s' % bd_id)
            self.updates_cmd.append('vxlan vni %s' % vni_id)
            self.changed = True

    def config_merge_mode(self, nve_name, mode):
        """config nve mode"""
        if self.is_nve_mode_change(nve_name, mode):
            cfg_xml = CE_NC_MERGE_NVE_MODE % (nve_name, mode)
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'MERGE_MODE')
            self.updates_cmd.append('interface %s' % nve_name)
            if mode == 'mode-l3':
                self.updates_cmd.append('mode l3')
            else:
                self.updates_cmd.append('undo mode l3')
            self.changed = True

    def config_merge_source_ip(self, nve_name, source_ip):
        """config nve source ip"""
        if self.is_nve_source_ip_change(nve_name, source_ip):
            cfg_xml = CE_NC_MERGE_NVE_SOURCE_IP_PROTOCOL % (nve_name, source_ip)
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'MERGE_SOURCE_IP')
            self.updates_cmd.append('interface %s' % nve_name)
            self.updates_cmd.append('source %s' % source_ip)
            self.changed = True

    def config_merge_vni_peer_ip(self, nve_name, vni_id, peer_ip_list):
        """config vni peer ip"""
        if self.is_vni_peer_list_change(nve_name, vni_id, peer_ip_list):
            cfg_xml = CE_NC_MERGE_VNI_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
            for peer_ip in peer_ip_list:
                cfg_xml += CE_NC_MERGE_VNI_PEER_ADDRESS_IP_MERGE % peer_ip
            cfg_xml += CE_NC_MERGE_VNI_PEER_ADDRESS_IP_END
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'MERGE_VNI_PEER_IP')
            self.updates_cmd.append('interface %s' % nve_name)
            for peer_ip in peer_ip_list:
                cmd_output = 'vni %s head-end peer-list %s' % (vni_id, peer_ip)
                self.updates_cmd.append(cmd_output)
            self.changed = True

    def config_merge_vni_protocol_type(self, nve_name, vni_id, protocol_type):
        """config vni protocol type"""
        if self.is_vni_protocol_change(nve_name, vni_id, protocol_type):
            cfg_xml = CE_NC_MERGE_VNI_PROTOCOL % (nve_name, vni_id, protocol_type)
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'MERGE_VNI_PEER_PROTOCOL')
            self.updates_cmd.append('interface %s' % nve_name)
            if protocol_type == 'bgp':
                self.updates_cmd.append('vni %s head-end peer-list protocol %s' % (vni_id, protocol_type))
            else:
                self.updates_cmd.append('undo vni %s head-end peer-list protocol bgp' % vni_id)
            self.changed = True

    def config_delete_vni2bd(self, bd_id, vni_id):
        """remove vni to bd id"""
        if not self.is_vni_bd_exist(vni_id, bd_id):
            return
        cfg_xml = CE_NC_DELETE_VNI_BD_ID % (vni_id, bd_id)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'DELETE_VNI_BD')
        self.updates_cmd.append('bridge-domain %s' % bd_id)
        self.updates_cmd.append('undo vxlan vni %s' % vni_id)
        self.changed = True

    def config_delete_mode(self, nve_name, mode):
        """nve mode"""
        if mode == 'mode-l3':
            if not self.is_nve_mode_exist(nve_name, mode):
                return
            cfg_xml = CE_NC_MERGE_NVE_MODE % (nve_name, 'mode-l2')
            recv_xml = set_nc_config(self.module, cfg_xml)
            self.check_response(recv_xml, 'DELETE_MODE')
            self.updates_cmd.append('interface %s' % nve_name)
            self.updates_cmd.append('undo mode l3')
            self.changed = True
        else:
            self.module.fail_json(msg='Error: Can not configure undo mode l2.')

    def config_delete_source_ip(self, nve_name, source_ip):
        """nve source ip"""
        if not self.is_nve_source_ip_exist(nve_name, source_ip):
            return
        ipaddr = '0.0.0.0'
        cfg_xml = CE_NC_MERGE_NVE_SOURCE_IP_PROTOCOL % (nve_name, ipaddr)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'DELETE_SOURCE_IP')
        self.updates_cmd.append('interface %s' % nve_name)
        self.updates_cmd.append('undo source %s' % source_ip)
        self.changed = True

    def config_delete_vni_peer_ip(self, nve_name, vni_id, peer_ip_list):
        """remove vni peer ip"""
        for peer_ip in peer_ip_list:
            if not self.is_vni_peer_list_exist(nve_name, vni_id, peer_ip):
                self.module.fail_json(msg='Error: The %s does not exist' % peer_ip)
        config = False
        nve_peer_info = list()
        for nve_peer in self.nve_info['vni_peer_ips']:
            if nve_peer['vniId'] == vni_id:
                nve_peer_info = nve_peer.get('peerAddr')
        for peer in nve_peer_info:
            if peer not in peer_ip_list:
                config = True
        if not config:
            cfg_xml = CE_NC_DELETE_VNI_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
            for peer_ip in peer_ip_list:
                cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_DELETE % peer_ip
            cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_END
        else:
            cfg_xml = CE_NC_DELETE_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
            for peer_ip in peer_ip_list:
                cfg_xml += CE_NC_DELETE_VNI_PEER_ADDRESS_IP_DELETE % peer_ip
            cfg_xml += CE_NC_DELETE_PEER_ADDRESS_IP_END
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'DELETE_VNI_PEER_IP')
        self.updates_cmd.append('interface %s' % nve_name)
        for peer_ip in peer_ip_list:
            cmd_output = 'undo vni %s head-end peer-list %s' % (vni_id, peer_ip)
            self.updates_cmd.append(cmd_output)
        self.changed = True

    def config_delete_vni_protocol_type(self, nve_name, vni_id, protocol_type):
        """remove vni protocol type"""
        if not self.is_vni_protocol_exist(nve_name, vni_id, protocol_type):
            return
        cfg_xml = CE_NC_DELETE_VNI_PROTOCOL % (nve_name, vni_id, protocol_type)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'DELETE_VNI_PEER_PROTOCOL')
        self.updates_cmd.append('interface %s' % nve_name)
        self.updates_cmd.append('undo vni %s head-end peer-list protocol bgp ' % vni_id)
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if self.bridge_domain_id:
            if not self.bridge_domain_id.isdigit():
                self.module.fail_json(msg='Error: The parameter of bridge domain id is invalid.')
            if int(self.bridge_domain_id) > 16777215 or int(self.bridge_domain_id) < 1:
                self.module.fail_json(msg='Error: The bridge domain id must be an integer between 1 and 16777215.')
        if self.vni_id:
            if not self.vni_id.isdigit():
                self.module.fail_json(msg='Error: The parameter of vni id is invalid.')
            if int(self.vni_id) > 16000000 or int(self.vni_id) < 1:
                self.module.fail_json(msg='Error: The vni id must be an integer between 1 and 16000000.')
        if self.nve_name:
            if not self.check_nve_name():
                self.module.fail_json(msg='Error: Error: NVE interface %s is invalid.' % self.nve_name)
        if self.peer_list_ip:
            for peer_ip in self.peer_list_ip:
                if not is_valid_address(peer_ip):
                    self.module.fail_json(msg='Error: The ip address %s is invalid.' % self.peer_list_ip)
        if self.source_ip:
            if not is_valid_address(self.source_ip):
                self.module.fail_json(msg='Error: The ip address %s is invalid.' % self.source_ip)

    def get_proposed(self):
        """get proposed info"""
        if self.bridge_domain_id:
            self.proposed['bridge_domain_id'] = self.bridge_domain_id
        if self.vni_id:
            self.proposed['vni_id'] = self.vni_id
        if self.nve_name:
            self.proposed['nve_name'] = self.nve_name
        if self.nve_mode:
            self.proposed['nve_mode'] = self.nve_mode
        if self.peer_list_ip:
            self.proposed['peer_list_ip'] = self.peer_list_ip
        if self.source_ip:
            self.proposed['source_ip'] = self.source_ip
        if self.state:
            self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if self.vni2bd_info:
            self.existing['vni_to_bridge_domain'] = self.vni2bd_info['vni2BdInfos']
        if self.nve_info:
            self.existing['nve_interface_name'] = self.nve_info['ifName']
            self.existing['source_ip'] = self.nve_info['srcAddr']
            self.existing['nve_mode'] = self.nve_info['nveType']
            self.existing['vni_peer_list_ip'] = self.nve_info['vni_peer_ips']
            self.existing['vni_peer_list_protocol'] = self.nve_info['vni_peer_protocols']

    def get_end_state(self):
        """get end state info"""
        vni2bd_info = self.get_vni2bd_dict()
        if vni2bd_info:
            self.end_state['vni_to_bridge_domain'] = vni2bd_info['vni2BdInfos']
        nve_info = self.get_nve_dict(self.nve_name)
        if nve_info:
            self.end_state['nve_interface_name'] = nve_info['ifName']
            self.end_state['source_ip'] = nve_info['srcAddr']
            self.end_state['nve_mode'] = nve_info['nveType']
            self.end_state['vni_peer_list_ip'] = nve_info['vni_peer_ips']
            self.end_state['vni_peer_list_protocol'] = nve_info['vni_peer_protocols']

    def work(self):
        """worker"""
        self.check_params()
        self.vni2bd_info = self.get_vni2bd_dict()
        if self.nve_name:
            self.nve_info = self.get_nve_dict(self.nve_name)
        self.get_existing()
        self.get_proposed()
        if self.state == 'present':
            if self.bridge_domain_id and self.vni_id:
                self.config_merge_vni2bd(self.bridge_domain_id, self.vni_id)
            if self.nve_name:
                if self.check_nve_interface(self.nve_name):
                    if self.nve_mode:
                        self.config_merge_mode(self.nve_name, self.nve_mode)
                    if self.source_ip:
                        self.config_merge_source_ip(self.nve_name, self.source_ip)
                    if self.vni_id and self.peer_list_ip:
                        self.config_merge_vni_peer_ip(self.nve_name, self.vni_id, self.peer_list_ip)
                    if self.vni_id and self.protocol_type:
                        self.config_merge_vni_protocol_type(self.nve_name, self.vni_id, self.protocol_type)
                else:
                    self.module.fail_json(msg='Error: Nve interface %s does not exist.' % self.nve_name)
        else:
            if self.bridge_domain_id and self.vni_id:
                self.config_delete_vni2bd(self.bridge_domain_id, self.vni_id)
            if self.nve_name:
                if self.check_nve_interface(self.nve_name):
                    if self.nve_mode:
                        self.config_delete_mode(self.nve_name, self.nve_mode)
                    if self.source_ip:
                        self.config_delete_source_ip(self.nve_name, self.source_ip)
                    if self.vni_id and self.peer_list_ip:
                        self.config_delete_vni_peer_ip(self.nve_name, self.vni_id, self.peer_list_ip)
                    if self.vni_id and self.protocol_type:
                        self.config_delete_vni_protocol_type(self.nve_name, self.vni_id, self.protocol_type)
                else:
                    self.module.fail_json(msg='Error: Nve interface %s does not exist.' % self.nve_name)
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)