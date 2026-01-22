from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class StaticRoute(object):
    """static route module"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.prefix = self.module.params['prefix']
        self.mask = self.module.params['mask']
        self.aftype = self.module.params['aftype']
        self.next_hop = self.module.params['next_hop']
        self.nhp_interface = self.module.params['nhp_interface']
        if self.nhp_interface is None:
            self.nhp_interface = 'Invalid0'
        self.tag = self.module.params['tag']
        self.description = self.module.params['description']
        self.state = self.module.params['state']
        self.pref = self.module.params['pref']
        self.vrf = self.module.params['vrf']
        if self.vrf is None:
            self.vrf = '_public_'
        self.destvrf = self.module.params['destvrf']
        if self.destvrf is None:
            self.destvrf = '_public_'
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.static_routes_info = dict()

    def init_module(self):
        """init module"""
        required_one_of = [['next_hop', 'nhp_interface']]
        self.module = AnsibleModule(argument_spec=self.spec, required_one_of=required_one_of, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def convert_len_to_mask(self, masklen):
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

    def convert_ip_prefix(self):
        """convert prefix to real value i.e. 2.2.2.2/24 to 2.2.2.0/24"""
        if self.aftype == 'v4':
            if self.prefix.find('.') == -1:
                return False
            if self.mask == '32':
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
        else:
            ipv6_addr_str = ''
            for num in range(0, length - ip_len):
                ipv6_addr_str += '%s:' % addr_list[num]
            self.prefix = ipv6_addr_str
            return True

    def set_update_cmd(self):
        """set update command"""
        if not self.changed:
            return
        if self.aftype == 'v4':
            aftype = 'ip'
            maskstr = self.convert_len_to_mask(self.mask)
        else:
            aftype = 'ipv6'
            maskstr = self.mask
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
                    self.updates_cmd.append('%s route-static vpn-instance %s %s %s vpn-instance %s %s' % (aftype, vrf, self.prefix, maskstr, destvrf, next_hop))
                else:
                    self.updates_cmd.append('%s route-static vpn-instance %s %s %s %s %s' % (aftype, vrf, self.prefix, maskstr, nhp_interface, next_hop))
            elif self.destvrf != '_public_':
                self.updates_cmd.append('%s route-static %s %s vpn-instance %s %s' % (aftype, self.prefix, maskstr, self.destvrf, next_hop))
            else:
                self.updates_cmd.append('%s route-static %s %s %s %s' % (aftype, self.prefix, maskstr, nhp_interface, next_hop))
            if self.pref:
                self.updates_cmd[0] += ' preference %s' % self.pref
            if self.tag:
                self.updates_cmd[0] += ' tag %s' % self.tag
            if self.description:
                self.updates_cmd[0] += ' description %s' % self.description
        if self.state == 'absent':
            if self.vrf != '_public_':
                if self.destvrf != '_public_':
                    self.updates_cmd.append('undo %s route-static vpn-instance %s %s %s vpn-instance %s %s' % (aftype, vrf, self.prefix, maskstr, destvrf, next_hop))
                else:
                    self.updates_cmd.append('undo %s route-static vpn-instance %s %s %s %s %s' % (aftype, vrf, self.prefix, maskstr, nhp_interface, next_hop))
            elif self.destvrf != '_public_':
                self.updates_cmd.append('undo %s route-static %s %s vpn-instance %s %s' % (aftype, self.prefix, maskstr, self.destvrf, self.next_hop))
            else:
                self.updates_cmd.append('undo %s route-static %s %s %s %s' % (aftype, self.prefix, maskstr, nhp_interface, next_hop))

    def operate_static_route(self, version, prefix, mask, nhp_interface, next_hop, vrf, destvrf, state):
        """operate ipv4 static route"""
        description_xml = '\n'
        preference_xml = '\n'
        tag_xml = '\n'
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
        if self.description:
            description_xml = CE_NC_SET_DESCRIPTION % self.description
        if self.pref:
            preference_xml = CE_NC_SET_PREFERENCE % self.pref
        if self.tag:
            tag_xml = CE_NC_SET_TAG % self.tag
        if state == 'present':
            configxmlstr = CE_NC_SET_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop, description_xml, preference_xml, tag_xml)
        else:
            configxmlstr = CE_NC_DELETE_STATIC_ROUTE % (vpn_instance, version, prefix, mask, nhp_interface, dest_vpn_instance, next_hop)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self.check_response(recv_xml, 'OPERATE_STATIC_ROUTE')

    def get_static_route(self, state):
        """get ipv4 static route"""
        self.static_routes_info['sroute'] = list()
        if state == 'absent':
            getxmlstr = CE_NC_GET_STATIC_ROUTE_ABSENT
        else:
            getxmlstr = CE_NC_GET_STATIC_ROUTE
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
                self.static_routes_info['sroute'].append(static_info)

    def check_params(self):
        """check all input params"""
        if not self.mask.isdigit():
            self.module.fail_json(msg='Error: Mask is invalid.')
        if self.aftype == 'v4':
            if int(self.mask) > 32 or int(self.mask) < 0:
                self.module.fail_json(msg='Error: Ipv4 mask must be an integer between 1 and 32.')
            if self.next_hop:
                if not is_valid_v4addr(self.next_hop):
                    self.module.fail_json(msg='Error: The %s is not a valid address' % self.next_hop)
        if self.aftype == 'v6':
            if int(self.mask) > 128 or int(self.mask) < 0:
                self.module.fail_json(msg='Error: Ipv6 mask must be an integer between 1 and 128.')
            if self.next_hop:
                if not is_valid_v6addr(self.next_hop):
                    self.module.fail_json(msg='Error: The %s is not a valid address' % self.next_hop)
        if self.description:
            if not is_valid_description(self.description):
                self.module.fail_json(msg='Error: Dsecription length should be 1 - 35, and can not contain "?".')
        if self.tag:
            if not is_valid_tag(self.tag):
                self.module.fail_json(msg='Error: Tag should be integer 1 - 4294967295.')
        if self.pref:
            if not is_valid_preference(self.pref):
                self.module.fail_json(msg='Error: Preference should be integer 1 - 255.')
        if self.nhp_interface != 'Invalid0' and self.destvrf != '_public_':
            self.module.fail_json(msg='Error: Destination vrf dose no support next hop is interface.')
        if not self.convert_ip_prefix():
            self.module.fail_json(msg='Error: The %s is not a valid address' % self.prefix)

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
        if self.aftype == 'v4':
            version = 'ipv4unicast'
        else:
            version = 'ipv6unicast'
        change = False
        self.get_static_route(self.state)
        if self.state == 'present':
            for static_route in self.static_routes_info['sroute']:
                if self.is_prefix_exist(static_route, version):
                    if self.vrf:
                        if static_route['vrfName'] != self.vrf:
                            change = True
                    if self.tag:
                        if static_route['tag'] != self.tag:
                            change = True
                    if self.destvrf:
                        if static_route['destVrfName'] != self.destvrf:
                            change = True
                    if self.description:
                        if static_route['description'] != self.description:
                            change = True
                    if self.pref:
                        if static_route['preference'] != self.pref:
                            change = True
                    if self.nhp_interface:
                        if static_route['ifName'].lower() != self.nhp_interface.lower():
                            change = True
                    if self.next_hop:
                        if static_route['nexthop'].lower() != self.next_hop.lower():
                            change = True
                    return change
                else:
                    continue
            change = True
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
        self.proposed['prefix'] = self.prefix
        self.proposed['mask'] = self.mask
        self.proposed['afType'] = self.aftype
        self.proposed['next_hop'] = self.next_hop
        self.proposed['ifName'] = self.nhp_interface
        self.proposed['vrfName'] = self.vrf
        self.proposed['destVrfName'] = self.destvrf
        if self.tag:
            self.proposed['tag'] = self.tag
        if self.description:
            self.proposed['description'] = self.description
        if self.pref is None:
            self.proposed['preference'] = 60
        else:
            self.proposed['preference'] = self.pref
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing information"""
        change = self.get_ip_static_route()
        self.existing['sroute'] = self.static_routes_info['sroute']
        self.changed = bool(change)

    def get_end_state(self):
        """get end state information"""
        self.get_static_route(self.state)
        self.end_state['sroute'] = self.static_routes_info['sroute']
        if self.end_state == self.existing:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        self.get_existing()
        self.get_proposed()
        self.set_ip_static_route()
        self.set_update_cmd()
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