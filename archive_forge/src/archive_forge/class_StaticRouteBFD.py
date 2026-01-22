from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
class StaticRouteBFD(object):
    """static route module"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self._initmodule_()
        self.function_flag = self.module.params['function_flag']
        self.aftype = self.module.params['aftype']
        self.state = self.module.params['state']
        if self.aftype == 'v4':
            self.version = 'ipv4unicast'
        else:
            self.version = 'ipv6unicast'
        if self.function_flag != 'globalBFD':
            self.nhp_interface = self.module.params['nhp_interface']
            if self.nhp_interface is None:
                self.nhp_interface = 'Invalid0'
            self.destvrf = self.module.params['destvrf']
            if self.destvrf is None:
                self.destvrf = '_public_'
            self.next_hop = self.module.params['next_hop']
            self.prefix = self.module.params['prefix']
        if self.function_flag != 'globalBFD' and self.function_flag != 'singleBFD':
            self.mask = self.module.params['mask']
            self.tag = self.module.params['tag']
            self.description = self.module.params['description']
            self.pref = self.module.params['pref']
            if self.pref is None:
                self.pref = 60
            self.vrf = self.module.params['vrf']
            if self.vrf is None:
                self.vrf = '_public_'
            self.bfd_session_name = self.module.params['bfd_session_name']
        if self.function_flag == 'globalBFD' or self.function_flag == 'singleBFD':
            self.min_tx_interval = self.module.params['min_tx_interval']
            self.min_rx_interval = self.module.params['min_rx_interval']
            self.detect_multiplier = self.module.params['detect_multiplier']
        if self.function_flag == 'globalBFD' and self.state == 'absent':
            self.commands = self.module.params['commands']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.static_routes_info = dict()

    def _initmodule_(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=False)

    def _checkresponse_(self, xml_str, xml_name):
        """check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def _convertlentomask_(self, masklen):
        """convert mask length to ip address mask, i.e. 24 to 255.255.255.0"""
        mask_int = ['0'] * 4
        length = int(masklen)
        if length > 32:
            self.module.fail_json(msg='IPv4 ipaddress mask length is invalid')
        if length < 8:
            mask_int[0] = str(int(255 << 8 - length % 8 & 255))
        if length >= 8:
            mask_int[0] = '255'
            mask_int[1] = str(int(255 << 16 - length % 16 & 255))
        if length >= 16:
            mask_int[1] = '255'
            mask_int[2] = str(int(255 << 24 - length % 24 & 255))
        if length >= 24:
            mask_int[2] = '255'
            mask_int[3] = str(int(255 << 32 - length % 32 & 255))
        if length == 32:
            mask_int[3] = '255'
        return '.'.join(mask_int)

    def _convertipprefix_(self):
        """convert prefix to real value i.e. 2.2.2.2/24 to 2.2.2.0/24"""
        if self.function_flag == 'singleBFD':
            if self.aftype == 'v4':
                if self.prefix.find('.') == -1:
                    return False
                addr_list = self.prefix.split('.')
                length = len(addr_list)
                if length > 4:
                    return False
                for each_num in addr_list:
                    if not each_num.isdigit():
                        return False
                    if int(each_num) > 255:
                        return False
                return True
            elif self.prefix.find(':') == -1:
                return False
        else:
            if self.aftype == 'v4':
                if self.prefix.find('.') == -1:
                    return False
                if self.mask == '32':
                    self.prefix = self.prefix
                    return True
                if self.mask == '0':
                    self.prefix = '0.0.0.0'
                    return True
                addr_list = self.prefix.split('.')
                length = len(addr_list)
                if length > 4:
                    return False
                for each_num in addr_list:
                    if not each_num.isdigit():
                        return False
                    if int(each_num) > 255:
                        return False
                byte_len = 8
                ip_len = int(self.mask) // byte_len
                ip_bit = int(self.mask) % byte_len
            else:
                if self.prefix.find(':') == -1:
                    return False
                if self.mask == '128':
                    self.prefix = self.prefix
                    return True
                if self.mask == '0':
                    self.prefix = '::'
                    return True
                addr_list = self.prefix.split(':')
                length = len(addr_list)
                if length > 6:
                    return False
                byte_len = 16
                ip_len = int(self.mask) // byte_len
                ip_bit = int(self.mask) % byte_len
            if self.aftype == 'v4':
                for i in range(ip_len + 1, length):
                    addr_list[i] = 0
            else:
                for i in range(length - ip_len, length):
                    addr_list[i] = 0
            for j in range(0, byte_len - ip_bit):
                if self.aftype == 'v4':
                    addr_list[ip_len] = int(addr_list[ip_len]) & 0 << j
                else:
                    if addr_list[length - ip_len - 1] == '':
                        continue
                    addr_list[length - ip_len - 1] = '0x%s' % addr_list[length - ip_len - 1]
                    addr_list[length - ip_len - 1] = int(addr_list[length - ip_len - 1], 16) & 0 << j
            if self.aftype == 'v4':
                self.prefix = '%s.%s.%s.%s' % (addr_list[0], addr_list[1], addr_list[2], addr_list[3])
                return True
            if self.aftype == 'v6':
                ipv6_addr_str = ''
                for num in range(0, length - ip_len):
                    ipv6_addr_str += '%s:' % addr_list[num]
                self.prefix = ipv6_addr_str
                return True

    def set_update_cmd_globalbfd(self):
        """set globalBFD update command"""
        if not self.changed:
            return
        if self.state == 'present':
            self.updates_cmd.append('ip route-static default-bfd')
            if self.min_tx_interval:
                self.updates_cmd.append(' min-rx-interval %s' % self.min_tx_interval)
            if self.min_rx_interval:
                self.updates_cmd.append(' min-tx-interval %s' % self.min_rx_interval)
            if self.detect_multiplier:
                self.updates_cmd.append(' detect-multiplier %s' % self.detect_multiplier)
        else:
            self.updates_cmd.append('undo ip route-static default-bfd')

    def set_update_cmd_singlebfd(self):
        """set singleBFD update command"""
        if not self.changed:
            return
        if self.next_hop is None:
            next_hop = ''
        else:
            next_hop = self.next_hop
        if self.destvrf == '_public_':
            destvrf = ''
        else:
            destvrf = self.destvrf
        if self.nhp_interface == 'Invalid0':
            nhp_interface = ''
        else:
            nhp_interface = self.nhp_interface
        if self.prefix == '0.0.0.0':
            prefix = ''
        else:
            prefix = self.prefix
        if self.state == 'present':
            if nhp_interface:
                self.updates_cmd.append('ip route-static bfd %s %s' % (nhp_interface, next_hop))
            elif destvrf:
                self.updates_cmd.append('ip route-static bfd vpn-instance %s %s' % (destvrf, next_hop))
            else:
                self.updates_cmd.append('ip route-static bfd %s' % next_hop)
            if prefix:
                self.updates_cmd.append(' local-address %s' % self.prefix)
            if self.min_tx_interval:
                self.updates_cmd.append(' min-rx-interval %s' % self.min_tx_interval)
            if self.min_rx_interval:
                self.updates_cmd.append(' min-tx-interval %s' % self.min_rx_interval)
            if self.detect_multiplier:
                self.updates_cmd.append(' detect-multiplier %s' % self.detect_multiplier)
        elif nhp_interface:
            self.updates_cmd.append('undo ip route-static bfd %s %s' % (nhp_interface, next_hop))
        elif destvrf:
            self.updates_cmd.append('undo ip route-static bfd vpn-instance %s %s' % (destvrf, next_hop))
        else:
            self.updates_cmd.append('undo ip route-static bfd %s' % next_hop)

    def set_update_cmd(self):
        """set update command"""
        if not self.changed:
            return
        if self.aftype == 'v4':
            maskstr = self._convertlentomask_(self.mask)
        else:
            maskstr = self.mask
        static_bfd_flag = True
        if self.bfd_session_name:
            static_bfd_flag = False
        if self.next_hop is None:
            next_hop = ''
        else:
            next_hop = self.next_hop
        if self.vrf == '_public_':
            vrf = ''
        else:
            vrf = self.vrf
        if self.destvrf == '_public_':
            destvrf = ''
        else:
            destvrf = self.destvrf
        if self.nhp_interface == 'Invalid0':
            nhp_interface = ''
        else:
            nhp_interface = self.nhp_interface
        if self.state == 'present':
            if self.vrf != '_public_':
                if self.destvrf != '_public_':
                    self.updates_cmd.append('ip route-static vpn-instance %s %s %s vpn-instance %s %s' % (vrf, self.prefix, maskstr, destvrf, next_hop))
                else:
                    self.updates_cmd.append('ip route-static vpn-instance %s %s %s %s %s' % (vrf, self.prefix, maskstr, nhp_interface, next_hop))
            elif self.destvrf != '_public_':
                self.updates_cmd.append('ip route-static %s %s vpn-instance %s %s' % (self.prefix, maskstr, self.destvrf, next_hop))
            else:
                self.updates_cmd.append('ip route-static %s %s %s %s' % (self.prefix, maskstr, nhp_interface, next_hop))
            if self.pref != 60:
                self.updates_cmd.append(' preference %s' % self.pref)
            if self.tag:
                self.updates_cmd.append(' tag %s' % self.tag)
            if not static_bfd_flag:
                self.updates_cmd.append(' track bfd-session %s' % self.bfd_session_name)
            else:
                self.updates_cmd.append(' bfd enable')
            if self.description:
                self.updates_cmd.append(' description %s' % self.description)
        if self.state == 'absent':
            if self.vrf != '_public_':
                if self.destvrf != '_public_':
                    self.updates_cmd.append('undo ip route-static vpn-instance %s %s %s vpn-instance %s %s' % (vrf, self.prefix, maskstr, destvrf, next_hop))
                else:
                    self.updates_cmd.append('undo ip route-static vpn-instance %s %s %s %s %s' % (vrf, self.prefix, maskstr, nhp_interface, next_hop))
            elif self.destvrf != '_public_':
                self.updates_cmd.append('undo ip route-static %s %s vpn-instance %s %s' % (self.prefix, maskstr, self.destvrf, self.next_hop))
            else:
                self.updates_cmd.append('undo ip route-static %s %s %s %s' % (self.prefix, maskstr, nhp_interface, next_hop))

    def operate_static_route_globalbfd(self):
        """set globalbfd update command"""
        min_tx_interval = self.min_tx_interval
        min_rx_interval = self.min_rx_interval
        multiplier = self.detect_multiplier
        min_tx_interval_xml = '\n'
        min_rx_interval_xml = '\n'
        multiplier_xml = '\n'
        if self.state == 'present':
            if min_tx_interval is not None:
                min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % min_tx_interval
            if min_rx_interval is not None:
                min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % min_rx_interval
            if multiplier is not None:
                multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % multiplier
            configxmlstr = CE_NC_SET_IPV4_STATIC_ROUTE_GLOBALBFD % (min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
            conf_str = build_config_xml(configxmlstr)
            recv_xml = set_nc_config(self.module, conf_str)
            self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_globalBFD')
        if self.state == 'absent' and self.commands:
            min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % 1000
            min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % 1000
            multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % 3
            configxmlstr = CE_NC_SET_IPV4_STATIC_ROUTE_GLOBALBFD % (min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
            conf_str = build_config_xml(configxmlstr)
            recv_xml = set_nc_config(self.module, conf_str)
            self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_globalBFD')

    def operate_static_route_singlebfd(self, version, prefix, nhp_interface, next_hop, destvrf, state):
        """operate ipv4 static route singleBFD"""
        min_tx_interval = self.min_tx_interval
        min_rx_interval = self.min_rx_interval
        multiplier = self.detect_multiplier
        min_tx_interval_xml = '\n'
        min_rx_interval_xml = '\n'
        multiplier_xml = '\n'
        local_address_xml = '\n'
        if next_hop is None:
            next_hop = '0.0.0.0'
        if destvrf is None:
            dest_vpn_instance = '_public_'
        else:
            dest_vpn_instance = destvrf
        if nhp_interface is None:
            nhp_interface = 'Invalid0'
        if min_tx_interval is not None:
            min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % min_tx_interval
        if min_rx_interval is not None:
            min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % min_rx_interval
        if multiplier is not None:
            multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % multiplier
        if prefix is not None:
            local_address_xml = CE_NC_SET_STATIC_ROUTE_SINGLEBFD_LOCALADRESS % prefix
        if state == 'present':
            configxmlstr = CE_NC_SET_STATIC_ROUTE_SINGLEBFD % (version, nhp_interface, dest_vpn_instance, next_hop, local_address_xml, min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
        else:
            configxmlstr = CE_NC_DELETE_STATIC_ROUTE_SINGLEBFD % (version, nhp_interface, dest_vpn_instance, next_hop)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_singleBFD')

    def operate_static_route(self, version, prefix, mask, nhp_interface, next_hop, vrf, destvrf, state):
        """operate ipv4 static route"""
        description_xml = '\n'
        preference_xml = '\n'
        tag_xml = '\n'
        bfd_xml = '\n'
        if next_hop is None:
            next_hop = '0.0.0.0'
        if nhp_interface is None:
            nhp_interface = 'Invalid0'
        if vrf is None:
            vpn_instance = '_public_'
        else:
            vpn_instance = vrf
        if destvrf is None:
            dest_vpn_instance = '_public_'
        else:
            dest_vpn_instance = destvrf
        description_xml = get_xml(CE_NC_SET_DESCRIPTION, self.description)
        preference_xml = get_xml(CE_NC_SET_PREFERENCE, self.pref)
        tag_xml = get_xml(CE_NC_SET_TAG, self.tag)
        if self.function_flag == 'staticBFD':
            if self.bfd_session_name:
                bfd_xml = CE_NC_SET_BFDSESSIONNAME % self.bfd_session_name
        else:
            bfd_xml = CE_NC_SET_BFDENABLE
        if state == 'present':
            configxmlstr = CE_NC_SET_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop, description_xml, preference_xml, tag_xml, bfd_xml)
        else:
            configxmlstr = CE_NC_DELETE_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE')

    def get_change_state_global_bfd(self):
        """get ipv4 global bfd change state"""
        self.get_global_bfd(self.state)
        change = False
        if self.state == 'present':
            if self.static_routes_info['sroute_global_bfd']:
                for static_route in self.static_routes_info['sroute_global_bfd']:
                    if static_route is not None:
                        if self.min_tx_interval is not None:
                            if int(static_route['minTxInterval']) != self.min_tx_interval:
                                change = True
                        if self.min_rx_interval is not None:
                            if int(static_route['minRxInterval']) != self.min_rx_interval:
                                change = True
                        if self.detect_multiplier is not None:
                            if int(static_route['multiplier']) != self.detect_multiplier:
                                change = True
                        return change
                    else:
                        continue
            else:
                change = True
        else:
            if self.commands:
                if self.static_routes_info['sroute_global_bfd']:
                    for static_route in self.static_routes_info['sroute_global_bfd']:
                        if static_route is not None:
                            if int(static_route['minTxInterval']) != 1000 or int(static_route['minRxInterval']) != 1000 or int(static_route['multiplier']) != 3:
                                change = True
            return change

    def get_global_bfd(self, state):
        """get ipv4 global bfd"""
        self.static_routes_info['sroute_global_bfd'] = list()
        getglobalbfdxmlstr = None
        if self.aftype == 'v4':
            getglobalbfdxmlstr = CE_NC_GET_STATIC_ROUTE_IPV4_GLOBAL_BFD
        if getglobalbfdxmlstr is not None:
            xml_global_bfd_str = get_nc_config(self.module, getglobalbfdxmlstr)
            if 'data/' in xml_global_bfd_str:
                return
            xml_global_bfd_str = xml_global_bfd_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_global_bfd_str)
            static_routes_global_bfd = root.findall('staticrt/staticrtbase/srIPv4StaticSite')
            if static_routes_global_bfd:
                for static_route in static_routes_global_bfd:
                    static_info = dict()
                    for static_ele in static_route:
                        if static_ele.tag == 'minTxInterval':
                            if static_ele.text is not None:
                                static_info['minTxInterval'] = static_ele.text
                        if static_ele.tag == 'minRxInterval':
                            if static_ele.text is not None:
                                static_info['minRxInterval'] = static_ele.text
                        if static_ele.tag == 'multiplier':
                            if static_ele.text is not None:
                                static_info['multiplier'] = static_ele.text
                    self.static_routes_info['sroute_global_bfd'].append(static_info)

    def get_change_state_single_bfd(self):
        """get ipv4 single bfd change state"""
        self.get_single_bfd(self.state)
        change = False
        version = self.version
        if self.state == 'present':
            if self.static_routes_info['sroute_single_bfd']:
                for static_route in self.static_routes_info['sroute_single_bfd']:
                    if static_route is not None and static_route['afType'] == version:
                        if self.nhp_interface:
                            if static_route['ifName'].lower() != self.nhp_interface.lower():
                                change = True
                        if self.destvrf:
                            if static_route['destVrfName'].lower() != self.destvrf.lower():
                                change = True
                        if self.next_hop:
                            if static_route['nexthop'].lower() != self.next_hop.lower():
                                change = True
                        if self.prefix:
                            if static_route['localAddress'].lower() != self.prefix.lower():
                                change = True
                        if self.min_tx_interval:
                            if int(static_route['minTxInterval']) != self.min_tx_interval:
                                change = True
                        if self.min_rx_interval:
                            if int(static_route['minRxInterval']) != self.min_rx_interval:
                                change = True
                        if self.detect_multiplier:
                            if int(static_route['multiplier']) != self.detect_multiplier:
                                change = True
                        return change
                    else:
                        continue
            else:
                change = True
        else:
            for static_route in self.static_routes_info['sroute_single_bfd']:
                if static_route['ifName'] and self.nhp_interface:
                    if static_route['ifName'].lower() == self.nhp_interface.lower() and static_route['nexthop'].lower() == self.next_hop.lower() and (static_route['afType'] == version):
                        change = True
                        return change
                if static_route['destVrfName'] and self.destvrf:
                    if static_route['destVrfName'].lower() == self.destvrf.lower() and static_route['nexthop'].lower() == self.next_hop.lower() and (static_route['afType'] == version):
                        change = True
                        return change
                if static_route['nexthop'] and self.next_hop:
                    if static_route['nexthop'].lower() == self.next_hop.lower() and static_route['afType'] == version:
                        change = True
                        return change
                else:
                    continue
            change = False
        return change

    def get_single_bfd(self, state):
        """get ipv4 sigle bfd"""
        self.static_routes_info['sroute_single_bfd'] = list()
        if self.aftype == 'v4':
            version = 'ipv4unicast'
        else:
            version = 'ipv6unicast'
        if state == 'absent':
            getbfdxmlstr = CE_NC_GET_STATIC_ROUTE_BFD_ABSENT % (version, self.nhp_interface, self.destvrf, self.next_hop)
        else:
            getbfdxmlstr = CE_NC_GET_STATIC_ROUTE_BFD % (version, self.nhp_interface, self.destvrf, self.next_hop)
        xml_bfd_str = get_nc_config(self.module, getbfdxmlstr)
        if 'data/' in xml_bfd_str:
            return
        xml_bfd_str = xml_bfd_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_bfd_str)
        static_routes_bfd = root.findall('staticrt/staticrtbase/srBfdParas/srBfdPara')
        if static_routes_bfd:
            for static_route in static_routes_bfd:
                static_info = dict()
                for static_ele in static_route:
                    if static_ele.tag in ['afType', 'destVrfName', 'nexthop', 'ifName']:
                        static_info[static_ele.tag] = static_ele.text
                    if static_ele.tag == 'localAddress':
                        if static_ele.text is not None:
                            static_info['localAddress'] = static_ele.text
                        else:
                            static_info['localAddress'] = 'None'
                    if static_ele.tag == 'minTxInterval':
                        if static_ele.text is not None:
                            static_info['minTxInterval'] = static_ele.text
                    if static_ele.tag == 'minRxInterval':
                        if static_ele.text is not None:
                            static_info['minRxInterval'] = static_ele.text
                    if static_ele.tag == 'multiplier':
                        if static_ele.text is not None:
                            static_info['multiplier'] = static_ele.text
                self.static_routes_info['sroute_single_bfd'].append(static_info)

    def get_static_route(self, state):
        """get ipv4 static route about BFD"""
        self.static_routes_info['sroute'] = list()
        static_bfd_flag = True
        if self.bfd_session_name:
            static_bfd_flag = False
        if state == 'absent':
            getxmlstr = CE_NC_GET_STATIC_ROUTE_ABSENT
        elif static_bfd_flag:
            getxmlstr = CE_NC_GET_STATIC_ROUTE_BFD_ENABLE
        else:
            getxmlstr = CE_NC_GET_STATIC_ROUTE_BFD_SESSIONNAME
        xml_str = get_nc_config(self.module, getxmlstr)
        if 'data/' in xml_str:
            return
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        static_routes = root.findall('staticrt/staticrtbase/srRoutes/srRoute')
        if static_routes:
            for static_route in static_routes:
                static_info = dict()
                for static_ele in static_route:
                    if static_ele.tag in ['vrfName', 'afType', 'topologyName', 'prefix', 'maskLength', 'destVrfName', 'nexthop', 'ifName', 'preference', 'description']:
                        static_info[static_ele.tag] = static_ele.text
                    if static_ele.tag == 'tag':
                        if static_ele.text is not None:
                            static_info['tag'] = static_ele.text
                        else:
                            static_info['tag'] = 'None'
                    if static_bfd_flag:
                        if static_ele.tag == 'bfdEnable':
                            if static_ele.text is not None:
                                static_info['bfdEnable'] = static_ele.text
                            else:
                                static_info['bfdEnable'] = 'None'
                    elif static_ele.tag == 'sessionName':
                        if static_ele.text is not None:
                            static_info['sessionName'] = static_ele.text
                        else:
                            static_info['sessionName'] = 'None'
                self.static_routes_info['sroute'].append(static_info)

    def _checkparams_(self):
        """check all input params"""
        if self.function_flag == 'singleBFD':
            if not self.next_hop:
                self.module.fail_json(msg='Error: missing required argument: next_hop.')
            if self.state != 'absent':
                if self.nhp_interface == 'Invalid0' and (not self.prefix or self.prefix == '0.0.0.0'):
                    self.module.fail_json(msg='Error: If a nhp_interface is not configured, the prefix must be configured.')
        if self.function_flag != 'globalBFD':
            if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
                if not self.mask:
                    self.module.fail_json(msg='Error: missing required argument: mask.')
                if not self.mask.isdigit():
                    self.module.fail_json(msg='Error: Mask is invalid.')
            if self.function_flag != 'singleBFD' or (self.function_flag == 'singleBFD' and self.destvrf != '_public_'):
                if not self.prefix:
                    self.module.fail_json(msg='Error: missing required argument: prefix.')
                if not self._convertipprefix_():
                    self.module.fail_json(msg='Error: The %s is not a valid address' % self.prefix)
            if self.nhp_interface != 'Invalid0' and self.destvrf != '_public_':
                self.module.fail_json(msg='Error: Destination vrf dose not support next hop is interface.')
            if not self.next_hop and self.nhp_interface == 'Invalid0':
                self.module.fail_json(msg='Error: one of the following is required: next_hop,nhp_interface.')
        if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
            if self.description:
                if not is_valid_description(self.description):
                    self.module.fail_json(msg='Error: Dsecription length should be 1 - 35, and can not contain "?".')
            if self.tag is not None:
                if not is_valid_tag(self.tag):
                    self.module.fail_json(msg='Error: Tag should be integer 1 - 4294967295.')
            if self.pref is not None:
                if not is_valid_preference(self.pref):
                    self.module.fail_json(msg='Error: Preference should be integer 1 - 255.')
            if self.function_flag == 'staticBFD':
                if self.bfd_session_name:
                    if not is_valid_bdf_session_name(self.bfd_session_name):
                        self.module.fail_json(msg='Error: bfd_session_name length should be 1 - 15, and can not contain Space.')
        if self.aftype == 'v4':
            if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
                if int(self.mask) > 32 or int(self.mask) < 0:
                    self.module.fail_json(msg='Error: Ipv4 mask must be an integer between 1 and 32.')
            if self.function_flag != 'globalBFD':
                if self.next_hop:
                    if not is_valid_v4addr(self.next_hop):
                        self.module.fail_json(msg='Error: The %s is not a valid address.' % self.next_hop)
        if self.aftype == 'v6':
            if self.function_flag == 'dynamicBFD' or self.function_flag == 'staticBFD':
                if int(self.mask) > 128 or int(self.mask) < 0:
                    self.module.fail_json(msg='Error: Ipv6 mask must be an integer between 1 and 128.')
            if self.function_flag != 'globalBFD':
                if self.next_hop:
                    if not is_valid_v6addr(self.next_hop):
                        self.module.fail_json(msg='Error: The %s is not a valid address.' % self.next_hop)
        if self.function_flag == 'globalBFD' or self.function_flag == 'singleBFD':
            if self.min_tx_interval:
                if not is_valid_bdf_interval(self.min_tx_interval):
                    self.module.fail_json(msg='Error: min_tx_interval should be integer 50 - 1000.')
            if self.min_rx_interval:
                if not is_valid_bdf_interval(self.min_rx_interval):
                    self.module.fail_json(msg='Error: min_rx_interval should be integer 50 - 1000.')
            if self.detect_multiplier:
                if not is_valid_bdf_multiplier(self.detect_multiplier):
                    self.module.fail_json(msg='Error: detect_multiplier should be integer 3 - 50.')
            if self.function_flag == 'globalBFD':
                if self.state != 'absent':
                    if not self.min_tx_interval and (not self.min_rx_interval) and (not self.detect_multiplier):
                        self.module.fail_json(msg='Error: one of the following is required: min_tx_interval,detect_multiplier,min_rx_interval.')
                else:
                    if not self.commands:
                        self.module.fail_json(msg='Error: missing required argument: command.')
                    if compare_command(self.commands):
                        self.module.fail_json(msg='Error: The command %s line is incorrect.' % ','.join(self.commands))

    def set_ip_static_route_globalbfd(self):
        """set ip static route globalBFD"""
        if not self.changed:
            return
        if self.aftype == 'v4':
            self.operate_static_route_globalbfd()

    def set_ip_static_route_singlebfd(self):
        """set ip static route singleBFD"""
        if not self.changed:
            return
        version = None
        if self.aftype == 'v4':
            version = 'ipv4unicast'
        else:
            version = 'ipv6unicast'
        self.operate_static_route_singlebfd(version, self.prefix, self.nhp_interface, self.next_hop, self.destvrf, self.state)

    def set_ip_static_route(self):
        """set ip static route"""
        if not self.changed:
            return
        version = None
        if self.aftype == 'v4':
            version = 'ipv4unicast'
        else:
            version = 'ipv6unicast'
        self.operate_static_route(version, self.prefix, self.mask, self.nhp_interface, self.next_hop, self.vrf, self.destvrf, self.state)

    def is_prefix_exist(self, static_route, version):
        """is prefix mask nex_thop exist"""
        if static_route is None:
            return False
        if self.next_hop and self.nhp_interface:
            return static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['afType'] == version) and (static_route['ifName'].lower() == self.nhp_interface.lower()) and (static_route['nexthop'].lower() == self.next_hop.lower())
        if self.next_hop and (not self.nhp_interface):
            return static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['afType'] == version) and (static_route['nexthop'].lower() == self.next_hop.lower())
        if not self.next_hop and self.nhp_interface:
            return static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['afType'] == version) and (static_route['ifName'].lower() == self.nhp_interface.lower())

    def get_ip_static_route(self):
        """get ip static route"""
        change = False
        version = self.version
        self.get_static_route(self.state)
        change_list = list()
        if self.state == 'present':
            for static_route in self.static_routes_info['sroute']:
                if self.is_prefix_exist(static_route, self.version):
                    info_dict = dict()
                    exist_dict = dict()
                    if self.vrf:
                        info_dict['vrfName'] = self.vrf
                        exist_dict['vrfName'] = static_route['vrfName']
                    if self.destvrf:
                        info_dict['destVrfName'] = self.destvrf
                        exist_dict['destVrfName'] = static_route['destVrfName']
                    if self.description:
                        info_dict['description'] = self.description
                        exist_dict['description'] = static_route['description']
                    if self.tag:
                        info_dict['tag'] = self.tag
                        exist_dict['tag'] = static_route['tag']
                    if self.pref:
                        info_dict['preference'] = str(self.pref)
                        exist_dict['preference'] = static_route['preference']
                    if self.nhp_interface:
                        if self.nhp_interface.lower() == 'invalid0':
                            info_dict['ifName'] = 'Invalid0'
                        else:
                            info_dict['ifName'] = 'Invalid0'
                        exist_dict['ifName'] = static_route['ifName']
                    if self.next_hop:
                        info_dict['nexthop'] = self.next_hop
                        exist_dict['nexthop'] = static_route['nexthop']
                    if self.bfd_session_name:
                        info_dict['bfdEnable'] = 'true'
                    else:
                        info_dict['bfdEnable'] = 'false'
                    exist_dict['bfdEnable'] = static_route['bfdEnable']
                    if exist_dict != info_dict:
                        change = True
                    else:
                        change = False
                    change_list.append(change)
            if False in change_list:
                change = False
            else:
                change = True
            return change
        else:
            for static_route in self.static_routes_info['sroute']:
                if static_route['nexthop'] and self.next_hop:
                    if static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['nexthop'].lower() == self.next_hop.lower()) and (static_route['afType'] == version):
                        change = True
                        return change
                if static_route['ifName'] and self.nhp_interface:
                    if static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['ifName'].lower() == self.nhp_interface.lower()) and (static_route['afType'] == version):
                        change = True
                        return change
                else:
                    continue
            change = False
        return change

    def get_proposed(self):
        """get proposed information"""
        self.proposed['afType'] = self.aftype
        self.proposed['state'] = self.state
        if self.function_flag != 'globalBFD':
            self.proposed['ifName'] = self.nhp_interface
            self.proposed['destVrfName'] = self.destvrf
            self.proposed['next_hop'] = self.next_hop
        if self.function_flag == 'singleBFD':
            if self.prefix:
                self.proposed['localAddress'] = self.prefix
        if self.function_flag == 'globalBFD' or self.function_flag == 'singleBFD':
            self.proposed['minTxInterval'] = self.min_tx_interval
            self.proposed['minRxInterval'] = self.min_rx_interval
            self.proposed['multiplier'] = self.detect_multiplier
        if self.function_flag != 'globalBFD' and self.function_flag != 'singleBFD':
            self.proposed['prefix'] = self.prefix
            self.proposed['mask'] = self.mask
            self.proposed['vrfName'] = self.vrf
            if self.tag:
                self.proposed['tag'] = self.tag
            if self.description:
                self.proposed['description'] = self.description
            if self.pref is None:
                self.proposed['preference'] = 60
            else:
                self.proposed['preference'] = self.pref
            static_bfd_flag = True
            if self.bfd_session_name:
                static_bfd_flag = False
            if not static_bfd_flag:
                self.proposed['sessionName'] = self.bfd_session_name
            else:
                self.proposed['bfdEnable'] = 'true'

    def get_existing(self):
        """get existing information"""
        if self.function_flag == 'globalBFD':
            change = self.get_change_state_global_bfd()
            self.existing['sroute_global_bfd'] = self.static_routes_info['sroute_global_bfd']
        elif self.function_flag == 'singleBFD':
            change = self.get_change_state_single_bfd()
            self.existing['sroute_single_bfd'] = self.static_routes_info['sroute_single_bfd']
        else:
            change = self.get_ip_static_route()
            self.existing['static_sroute'] = self.static_routes_info['sroute']
        self.changed = bool(change)

    def get_end_state(self):
        """get end state information"""
        if self.function_flag == 'globalBFD':
            self.get_global_bfd(self.state)
            self.end_state['sroute_global_bfd'] = self.static_routes_info['sroute_global_bfd']
        elif self.function_flag == 'singleBFD':
            self.static_routes_info['sroute_single_bfd'] = list()
            self.get_single_bfd(self.state)
            self.end_state['sroute_single_bfd'] = self.static_routes_info['sroute_single_bfd']
        else:
            self.get_static_route(self.state)
            self.end_state['static_sroute'] = self.static_routes_info['sroute']

    def work(self):
        """worker"""
        self._checkparams_()
        self.get_existing()
        self.get_proposed()
        if self.function_flag == 'globalBFD':
            self.set_ip_static_route_globalbfd()
            self.set_update_cmd_globalbfd()
        elif self.function_flag == 'singleBFD':
            self.set_ip_static_route_singlebfd()
            self.set_update_cmd_singlebfd()
        else:
            self.set_ip_static_route()
            self.set_update_cmd()
        self.get_end_state()
        if self.existing == self.end_state:
            self.changed = False
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)