from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class EvpnBd(object):
    """Manage evpn instance in BD view"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.bridge_domain_id = self.module.params['bridge_domain_id']
        self.evpn = self.module.params['evpn']
        self.route_distinguisher = self.module.params['route_distinguisher']
        self.vpn_target_both = self.module.params['vpn_target_both'] or list()
        self.vpn_target_import = self.module.params['vpn_target_import'] or list()
        self.vpn_target_export = self.module.params['vpn_target_export'] or list()
        self.state = self.module.params['state']
        self.__string_to_lowercase__()
        self.commands = list()
        self.evpn_info = dict()
        self.conf_exist = False
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def __init_module__(self):
        """Init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def __check_response__(self, xml_str, xml_name):
        """Check if response message is already succeed"""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def __string_to_lowercase__(self):
        """Convert string to lowercase"""
        if self.route_distinguisher:
            self.route_distinguisher = self.route_distinguisher.lower()
        if self.vpn_target_export:
            for index, ele in enumerate(self.vpn_target_export):
                self.vpn_target_export[index] = ele.lower()
        if self.vpn_target_import:
            for index, ele in enumerate(self.vpn_target_import):
                self.vpn_target_import[index] = ele.lower()
        if self.vpn_target_both:
            for index, ele in enumerate(self.vpn_target_both):
                self.vpn_target_both[index] = ele.lower()

    def get_all_evpn_rts(self, evpn_rts):
        """Get all EVPN RTS"""
        rts = evpn_rts.findall('evpnRT')
        if not rts:
            return
        for ele in rts:
            vrf_rttype = ele.find('vrfRTType')
            vrf_rtvalue = ele.find('vrfRTValue')
            if vrf_rttype.text == 'export_extcommunity':
                self.evpn_info['vpn_target_export'].append(vrf_rtvalue.text)
            elif vrf_rttype.text == 'import_extcommunity':
                self.evpn_info['vpn_target_import'].append(vrf_rtvalue.text)

    def get_all_evpn_autorts(self, evpn_autorts):
        """"Get all EVPN AUTORTS"""
        autorts = evpn_autorts.findall('evpnAutoRT')
        if not autorts:
            return
        for autort in autorts:
            vrf_rttype = autort.find('vrfRTType')
            if vrf_rttype.text == 'export_extcommunity':
                self.evpn_info['vpn_target_export'].append('auto')
            elif vrf_rttype.text == 'import_extcommunity':
                self.evpn_info['vpn_target_import'].append('auto')

    def process_rts_info(self):
        """Process RTS information"""
        if not self.evpn_info['vpn_target_export'] or not self.evpn_info['vpn_target_import']:
            return
        vpn_target_export = copy.deepcopy(self.evpn_info['vpn_target_export'])
        for ele in vpn_target_export:
            if ele in self.evpn_info['vpn_target_import']:
                self.evpn_info['vpn_target_both'].append(ele)
                self.evpn_info['vpn_target_export'].remove(ele)
                self.evpn_info['vpn_target_import'].remove(ele)

    def get_evpn_instance_info(self):
        """Get current EVPN instance information"""
        if not self.bridge_domain_id:
            self.module.fail_json(msg='Error: The value of bridge_domain_id cannot be empty.')
        self.evpn_info['route_distinguisher'] = None
        self.evpn_info['vpn_target_import'] = list()
        self.evpn_info['vpn_target_export'] = list()
        self.evpn_info['vpn_target_both'] = list()
        self.evpn_info['evpn_inst'] = 'enable'
        xml_str = CE_NC_GET_EVPN_CONFIG % (self.bridge_domain_id, self.bridge_domain_id)
        xml_str = get_nc_config(self.module, xml_str)
        if '<data/>' in xml_str:
            self.evpn_info['evpn_inst'] = 'disable'
            return
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        evpn_inst = root.find('evpn/evpnInstances/evpnInstance')
        if evpn_inst:
            for eles in evpn_inst:
                if eles.tag in ['evpnAutoRD', 'evpnRD', 'evpnRTs', 'evpnAutoRTs']:
                    if eles.tag == 'evpnAutoRD' and eles.text == 'true':
                        self.evpn_info['route_distinguisher'] = 'auto'
                    elif eles.tag == 'evpnRD' and self.evpn_info['route_distinguisher'] != 'auto':
                        self.evpn_info['route_distinguisher'] = eles.text
                    elif eles.tag == 'evpnRTs':
                        self.get_all_evpn_rts(eles)
                    elif eles.tag == 'evpnAutoRTs':
                        self.get_all_evpn_autorts(eles)
            self.process_rts_info()

    def get_existing(self):
        """Get existing config"""
        self.existing = dict(bridge_domain_id=self.bridge_domain_id, evpn=self.evpn_info['evpn_inst'], route_distinguisher=self.evpn_info['route_distinguisher'], vpn_target_both=self.evpn_info['vpn_target_both'], vpn_target_import=self.evpn_info['vpn_target_import'], vpn_target_export=self.evpn_info['vpn_target_export'])

    def get_proposed(self):
        """Get proposed config"""
        self.proposed = dict(bridge_domain_id=self.bridge_domain_id, evpn=self.evpn, route_distinguisher=self.route_distinguisher, vpn_target_both=self.vpn_target_both, vpn_target_import=self.vpn_target_import, vpn_target_export=self.vpn_target_export, state=self.state)

    def get_end_state(self):
        """Get end config"""
        self.get_evpn_instance_info()
        self.end_state = dict(bridge_domain_id=self.bridge_domain_id, evpn=self.evpn_info['evpn_inst'], route_distinguisher=self.evpn_info['route_distinguisher'], vpn_target_both=self.evpn_info['vpn_target_both'], vpn_target_import=self.evpn_info['vpn_target_import'], vpn_target_export=self.evpn_info['vpn_target_export'])

    def show_result(self):
        """Show result"""
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)

    def judge_if_vpn_target_exist(self, vpn_target_type):
        """Judge whether proposed vpn target has existed"""
        vpn_target = list()
        if vpn_target_type == 'vpn_target_import':
            vpn_target.extend(self.existing['vpn_target_both'])
            vpn_target.extend(self.existing['vpn_target_import'])
            return set(self.proposed['vpn_target_import']).issubset(vpn_target)
        elif vpn_target_type == 'vpn_target_export':
            vpn_target.extend(self.existing['vpn_target_both'])
            vpn_target.extend(self.existing['vpn_target_export'])
            return set(self.proposed['vpn_target_export']).issubset(vpn_target)
        return False

    def judge_if_config_exist(self):
        """Judge whether configuration has existed"""
        if self.state == 'absent':
            if self.route_distinguisher or self.vpn_target_import or self.vpn_target_export or self.vpn_target_both:
                return False
            else:
                return True
        if self.evpn_info['evpn_inst'] != self.evpn:
            return False
        if self.evpn == 'disable' and self.evpn_info['evpn_inst'] == 'disable':
            return True
        if self.proposed['bridge_domain_id'] != self.existing['bridge_domain_id']:
            return False
        if self.proposed['route_distinguisher']:
            if self.proposed['route_distinguisher'] != self.existing['route_distinguisher']:
                return False
        if self.proposed['vpn_target_both']:
            if not self.existing['vpn_target_both']:
                return False
            if not set(self.proposed['vpn_target_both']).issubset(self.existing['vpn_target_both']):
                return False
        if self.proposed['vpn_target_import']:
            if not self.judge_if_vpn_target_exist('vpn_target_import'):
                return False
        if self.proposed['vpn_target_export']:
            if not self.judge_if_vpn_target_exist('vpn_target_export'):
                return False
        return True

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

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

    def config_evpn_instance(self):
        """Configure EVPN instance"""
        self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
        if self.evpn == 'disable':
            xml_str = CE_NC_DELETE_EVPN_CONFIG % (self.bridge_domain_id, self.bridge_domain_id)
            recv_xml = set_nc_config(self.module, xml_str)
            self.check_response(recv_xml, 'MERGE_EVPN_BD_CONFIG')
            self.updates_cmd.append('  undo evpn')
            self.changed = True
            return
        xml_str = CE_NC_MERGE_EVPN_CONFIG_HEAD % (self.bridge_domain_id, self.bridge_domain_id)
        self.updates_cmd.append('  evpn')
        if self.route_distinguisher:
            if not self.existing['route_distinguisher']:
                if self.route_distinguisher.lower() == 'auto':
                    xml_str += '<evpnAutoRD>true</evpnAutoRD>'
                    self.updates_cmd.append('    route-distinguisher auto')
                else:
                    xml_str += '<evpnRD>%s</evpnRD>' % self.route_distinguisher
                    self.updates_cmd.append('    route-distinguisher %s' % self.route_distinguisher)
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
                if ele.lower() == 'auto' and (not self.is_vpn_target_exist('export_extcommunity', ele.lower())):
                    if not head_flag:
                        xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                        head_flag = True
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_CONTEXT % 'export_extcommunity'
                    self.updates_cmd.append('    vpn-target auto export-extcommunity')
        if vpn_target_import:
            for ele in vpn_target_import:
                if ele.lower() == 'auto' and (not self.is_vpn_target_exist('import_extcommunity', ele.lower())):
                    if not head_flag:
                        xml_str += CE_NC_MERGE_EVPN_AUTORTS_HEAD
                        head_flag = True
                    xml_str += CE_NC_MERGE_EVPN_AUTORTS_CONTEXT % 'import_extcommunity'
                    self.updates_cmd.append('    vpn-target auto import-extcommunity')
        if head_flag:
            xml_str += CE_NC_MERGE_EVPN_AUTORTS_TAIL
        head_flag = False
        if vpn_target_export:
            for ele in vpn_target_export:
                if ele.lower() != 'auto' and (not self.is_vpn_target_exist('export_extcommunity', ele.lower())):
                    if not head_flag:
                        xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                        head_flag = True
                    xml_str += CE_NC_MERGE_EVPN_RTS_CONTEXT % ('export_extcommunity', ele)
                    self.updates_cmd.append('    vpn-target %s export-extcommunity' % ele)
        if vpn_target_import:
            for ele in vpn_target_import:
                if ele.lower() != 'auto' and (not self.is_vpn_target_exist('import_extcommunity', ele.lower())):
                    if not head_flag:
                        xml_str += CE_NC_MERGE_EVPN_RTS_HEAD
                        head_flag = True
                    xml_str += CE_NC_MERGE_EVPN_RTS_CONTEXT % ('import_extcommunity', ele)
                    self.updates_cmd.append('    vpn-target %s import-extcommunity' % ele)
        if head_flag:
            xml_str += CE_NC_MERGE_EVPN_RTS_TAIL
        xml_str += CE_NC_MERGE_EVPN_CONFIG_TAIL
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'MERGE_EVPN_BD_CONFIG')
        self.changed = True

    def is_vpn_target_exist(self, target_type, value):
        """Judge whether VPN target has existed"""
        if target_type == 'export_extcommunity':
            if value not in self.existing['vpn_target_export'] and value not in self.existing['vpn_target_both']:
                return False
            return True
        if target_type == 'import_extcommunity':
            if value not in self.existing['vpn_target_import'] and value not in self.existing['vpn_target_both']:
                return False
            return True
        return False

    def config_evnp_bd(self):
        """Configure EVPN in BD view"""
        if not self.conf_exist:
            if self.state == 'present':
                self.config_evpn_instance()
            else:
                self.unconfig_evpn_instance()

    def process_input_params(self):
        """Process input parameters"""
        if self.state == 'absent':
            self.evpn = None
        elif self.evpn == 'disable':
            return
        if self.vpn_target_both:
            for ele in self.vpn_target_both:
                if ele in self.vpn_target_export:
                    self.vpn_target_export.remove(ele)
                if ele in self.vpn_target_import:
                    self.vpn_target_import.remove(ele)
        if self.vpn_target_export and self.vpn_target_import:
            vpn_target_export = copy.deepcopy(self.vpn_target_export)
            for ele in vpn_target_export:
                if ele in self.vpn_target_import:
                    self.vpn_target_both.append(ele)
                    self.vpn_target_import.remove(ele)
                    self.vpn_target_export.remove(ele)

    def check_vpn_target_para(self):
        """Check whether VPN target value is valid"""
        if self.route_distinguisher:
            if self.route_distinguisher.lower() != 'auto' and (not is_valid_value(self.route_distinguisher)):
                self.module.fail_json(msg='Error: Route distinguisher has invalid value %s.' % self.route_distinguisher)
        if self.vpn_target_export:
            for ele in self.vpn_target_export:
                if ele.lower() != 'auto' and (not is_valid_value(ele)):
                    self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)
        if self.vpn_target_import:
            for ele in self.vpn_target_import:
                if ele.lower() != 'auto' and (not is_valid_value(ele)):
                    self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)
        if self.vpn_target_both:
            for ele in self.vpn_target_both:
                if ele.lower() != 'auto' and (not is_valid_value(ele)):
                    self.module.fail_json(msg='Error: VPN target extended community attribute has invalid value %s.' % ele)

    def check_undo_params_if_exist(self):
        """Check whether all undo parameters is existed"""
        if self.vpn_target_import:
            for ele in self.vpn_target_import:
                if ele not in self.evpn_info['vpn_target_import'] and ele not in self.evpn_info['vpn_target_both']:
                    self.module.fail_json(msg='Error: VPN target import attribute value %s does not exist.' % ele)
        if self.vpn_target_export:
            for ele in self.vpn_target_export:
                if ele not in self.evpn_info['vpn_target_export'] and ele not in self.evpn_info['vpn_target_both']:
                    self.module.fail_json(msg='Error: VPN target export attribute value %s does not exist.' % ele)
        if self.vpn_target_both:
            for ele in self.vpn_target_both:
                if ele not in self.evpn_info['vpn_target_both']:
                    self.module.fail_json(msg='Error: VPN target export and import attribute value %s does not exist.' % ele)

    def check_params(self):
        """Check all input params"""
        if self.bridge_domain_id:
            if not self.bridge_domain_id.isdigit():
                self.module.fail_json(msg='Error: The parameter of bridge domain id is invalid.')
            if int(self.bridge_domain_id) > 16777215 or int(self.bridge_domain_id) < 1:
                self.module.fail_json(msg='Error: The bridge domain id must be an integer between 1 and 16777215.')
        if self.state == 'absent':
            self.check_undo_params_if_exist()
        self.check_vni_bd()
        self.check_vpn_target_para()
        if self.state == 'absent':
            if self.route_distinguisher:
                if not self.evpn_info['route_distinguisher']:
                    self.module.fail_json(msg='Error: Route distinguisher has not been configured.')
                elif self.route_distinguisher != self.evpn_info['route_distinguisher']:
                    self.module.fail_json(msg='Error: Current route distinguisher value is %s.' % self.evpn_info['route_distinguisher'])
        if self.state == 'present':
            if self.route_distinguisher:
                if self.evpn_info['route_distinguisher'] and self.route_distinguisher != self.evpn_info['route_distinguisher']:
                    self.module.fail_json(msg='Error: Route distinguisher has already been configured.')

    def check_vni_bd(self):
        """Check whether vxlan vni is configured in BD view"""
        xml_str = CE_NC_GET_VNI_BD
        xml_str = get_nc_config(self.module, xml_str)
        if '<data/>' in xml_str or not re.findall('<vniId>\\S+</vniId>\\s+<bdId>%s</bdId>' % self.bridge_domain_id, xml_str):
            self.module.fail_json(msg='Error: The vxlan vni is not configured or the bridge domain id is invalid.')

    def work(self):
        """Execute task"""
        self.get_evpn_instance_info()
        self.process_input_params()
        self.check_params()
        self.get_existing()
        self.get_proposed()
        self.conf_exist = self.judge_if_config_exist()
        self.config_evnp_bd()
        self.get_end_state()
        self.show_result()