from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class EthTrunk(object):
    """
    Manages Eth-Trunk interfaces.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.trunk_id = self.module.params['trunk_id']
        self.mode = self.module.params['mode']
        self.min_links = self.module.params['min_links']
        self.hash_type = self.module.params['hash_type']
        self.members = self.module.params['members']
        self.state = self.module.params['state']
        self.force = self.module.params['force']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.trunk_info = dict()

    def __init_module__(self):
        """ init module """
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def netconf_set_config(self, xml_str, xml_name):
        """ netconf set config """
        recv_xml = set_nc_config(self.module, xml_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_trunk_dict(self, trunk_id):
        """ get one interface attributes dict."""
        trunk_info = dict()
        conf_str = CE_NC_GET_TRUNK % trunk_id
        recv_xml = get_nc_config(self.module, conf_str)
        if '<data/>' in recv_xml:
            return trunk_info
        base = re.findall('.*<ifName>(.*)</ifName>.*\\s*<minUpNum>(.*)</minUpNum>.*\\s*<maxUpNum>(.*)</maxUpNum>.*\\s*<trunkType>(.*)</trunkType>.*\\s*<hashType>(.*)</hashType>.*\\s*<workMode>(.*)</workMode>.*\\s*<upMemberIfNum>(.*)</upMemberIfNum>.*\\s*<memberIfNum>(.*)</memberIfNum>.*', recv_xml)
        if base:
            trunk_info = dict(ifName=base[0][0], trunkId=base[0][0].lower().replace('eth-trunk', '').replace(' ', ''), minUpNum=base[0][1], maxUpNum=base[0][2], trunkType=base[0][3], hashType=base[0][4], workMode=base[0][5], upMemberIfNum=base[0][6], memberIfNum=base[0][7])
        member = re.findall('.*<memberIfName>(.*)</memberIfName>.*\\s*<memberIfState>(.*)</memberIfState>.*', recv_xml)
        trunk_info['TrunkMemberIfs'] = list()
        for mem in member:
            trunk_info['TrunkMemberIfs'].append(dict(memberIfName=mem[0], memberIfState=mem[1]))
        return trunk_info

    def is_member_exist(self, ifname):
        """is trunk member exist"""
        if not self.trunk_info['TrunkMemberIfs']:
            return False
        for mem in self.trunk_info['TrunkMemberIfs']:
            if ifname.replace(' ', '').upper() == mem['memberIfName'].replace(' ', '').upper():
                return True
        return False

    def get_mode_xml_str(self):
        """trunk mode netconf xml format string"""
        return MODE_CLI2XML.get(self.mode)

    def get_hash_type_xml_str(self):
        """trunk hash type netconf xml format string"""
        return HASH_CLI2XML.get(self.hash_type)

    def create_eth_trunk(self):
        """Create Eth-Trunk interface"""
        xml_str = CE_NC_XML_CREATE_TRUNK % self.trunk_id
        self.updates_cmd.append('interface Eth-Trunk %s' % self.trunk_id)
        if self.hash_type:
            self.updates_cmd.append('load-balance %s' % self.hash_type)
            xml_str += CE_NC_XML_MERGE_HASHTYPE % (self.trunk_id, self.get_hash_type_xml_str())
        if self.mode:
            self.updates_cmd.append('mode %s' % self.mode)
            xml_str += CE_NC_XML_MERGE_WORKMODE % (self.trunk_id, self.get_mode_xml_str())
        if self.min_links:
            self.updates_cmd.append('least active-linknumber %s' % self.min_links)
            xml_str += CE_NC_XML_MERGE_MINUPNUM % (self.trunk_id, self.min_links)
        if self.members:
            mem_xml = ''
            for mem in self.members:
                mem_xml += CE_NC_XML_MERGE_MEMBER % mem.upper()
                self.updates_cmd.append('interface %s' % mem)
                self.updates_cmd.append('eth-trunk %s' % self.trunk_id)
            xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
        cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
        self.netconf_set_config(cfg_xml, 'CREATE_TRUNK')
        self.changed = True

    def delete_eth_trunk(self):
        """Delete Eth-Trunk interface and remove all member"""
        if not self.trunk_info:
            return
        xml_str = ''
        mem_str = ''
        if self.trunk_info['TrunkMemberIfs']:
            for mem in self.trunk_info['TrunkMemberIfs']:
                mem_str += CE_NC_XML_DELETE_MEMBER % mem['memberIfName']
                self.updates_cmd.append('interface %s' % mem['memberIfName'])
                self.updates_cmd.append('undo eth-trunk')
            if mem_str:
                xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_str)
        xml_str += CE_NC_XML_DELETE_TRUNK % self.trunk_id
        self.updates_cmd.append('undo interface Eth-Trunk %s' % self.trunk_id)
        cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
        self.netconf_set_config(cfg_xml, 'DELETE_TRUNK')
        self.changed = True

    def remove_member(self):
        """delete trunk member"""
        if not self.members:
            return
        change = False
        mem_xml = ''
        xml_str = ''
        for mem in self.members:
            if self.is_member_exist(mem):
                mem_xml += CE_NC_XML_DELETE_MEMBER % mem.upper()
                self.updates_cmd.append('interface %s' % mem)
                self.updates_cmd.append('undo eth-trunk')
        if mem_xml:
            xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
            change = True
        if not change:
            return
        cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
        self.netconf_set_config(cfg_xml, 'REMOVE_TRUNK_MEMBER')
        self.changed = True

    def merge_eth_trunk(self):
        """Create or merge Eth-Trunk"""
        change = False
        xml_str = ''
        self.updates_cmd.append('interface Eth-Trunk %s' % self.trunk_id)
        if self.hash_type and self.get_hash_type_xml_str() != self.trunk_info['hashType']:
            self.updates_cmd.append('load-balance %s' % self.hash_type)
            xml_str += CE_NC_XML_MERGE_HASHTYPE % (self.trunk_id, self.get_hash_type_xml_str())
            change = True
        if self.min_links and self.min_links != self.trunk_info['minUpNum']:
            self.updates_cmd.append('least active-linknumber %s' % self.min_links)
            xml_str += CE_NC_XML_MERGE_MINUPNUM % (self.trunk_id, self.min_links)
            change = True
        if self.mode and self.get_mode_xml_str() != self.trunk_info['workMode']:
            self.updates_cmd.append('mode %s' % self.mode)
            xml_str += CE_NC_XML_MERGE_WORKMODE % (self.trunk_id, self.get_mode_xml_str())
            change = True
        if not change:
            self.updates_cmd.pop()
        if self.force and self.trunk_info['TrunkMemberIfs']:
            mem_xml = ''
            for mem in self.trunk_info['TrunkMemberIfs']:
                if not self.members or mem['memberIfName'].replace(' ', '').upper() not in self.members:
                    mem_xml += CE_NC_XML_DELETE_MEMBER % mem['memberIfName']
                    self.updates_cmd.append('interface %s' % mem['memberIfName'])
                    self.updates_cmd.append('undo eth-trunk')
            if mem_xml:
                xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
                change = True
        if self.members:
            mem_xml = ''
            for mem in self.members:
                if not self.is_member_exist(mem):
                    mem_xml += CE_NC_XML_MERGE_MEMBER % mem.upper()
                    self.updates_cmd.append('interface %s' % mem)
                    self.updates_cmd.append('eth-trunk %s' % self.trunk_id)
            if mem_xml:
                xml_str += CE_NC_XML_BUILD_MEMBER_CFG % (self.trunk_id, mem_xml)
                change = True
        if not change:
            return
        cfg_xml = CE_NC_XML_BUILD_TRUNK_CFG % xml_str
        self.netconf_set_config(cfg_xml, 'MERGE_TRUNK')
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if not self.trunk_id.isdigit():
            self.module.fail_json(msg='The parameter of trunk_id is invalid.')
        if self.min_links and (not self.min_links.isdigit()):
            self.module.fail_json(msg='The parameter of min_links is invalid.')
        if self.members:
            for mem in self.members:
                if not get_interface_type(mem.replace(' ', '')):
                    self.module.fail_json(msg='The parameter of members is invalid.')
            for mem_id in range(len(self.members)):
                self.members[mem_id] = self.members[mem_id].replace(' ', '').upper()

    def get_proposed(self):
        """get proposed info"""
        self.proposed['trunk_id'] = self.trunk_id
        self.proposed['mode'] = self.mode
        if self.min_links:
            self.proposed['min_links'] = self.min_links
        self.proposed['hash_type'] = self.hash_type
        if self.members:
            self.proposed['members'] = self.members
        self.proposed['state'] = self.state
        self.proposed['force'] = self.force

    def get_existing(self):
        """get existing info"""
        if not self.trunk_info:
            return
        self.existing['trunk_id'] = self.trunk_info['trunkId']
        self.existing['min_links'] = self.trunk_info['minUpNum']
        self.existing['hash_type'] = hash_type_xml_to_cli_str(self.trunk_info['hashType'])
        self.existing['mode'] = mode_xml_to_cli_str(self.trunk_info['workMode'])
        self.existing['members_detail'] = self.trunk_info['TrunkMemberIfs']

    def get_end_state(self):
        """get end state info"""
        trunk_info = self.get_trunk_dict(self.trunk_id)
        if not trunk_info:
            return
        self.end_state['trunk_id'] = trunk_info['trunkId']
        self.end_state['min_links'] = trunk_info['minUpNum']
        self.end_state['hash_type'] = hash_type_xml_to_cli_str(trunk_info['hashType'])
        self.end_state['mode'] = mode_xml_to_cli_str(trunk_info['workMode'])
        self.end_state['members_detail'] = trunk_info['TrunkMemberIfs']

    def work(self):
        """worker"""
        self.check_params()
        self.trunk_info = self.get_trunk_dict(self.trunk_id)
        self.get_existing()
        self.get_proposed()
        if self.state == 'present':
            if not self.trunk_info:
                self.create_eth_trunk()
            else:
                self.merge_eth_trunk()
        elif self.trunk_info:
            if not self.members:
                self.delete_eth_trunk()
            else:
                self.remove_member()
        else:
            self.module.fail_json(msg='Error: Eth-Trunk does not exist.')
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