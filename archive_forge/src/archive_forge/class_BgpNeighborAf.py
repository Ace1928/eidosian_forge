from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
class BgpNeighborAf(object):
    """ Manages BGP neighbor Address-family configuration """

    def netconf_get_config(self, **kwargs):
        """ netconf_get_config """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = get_nc_config(module, conf_str)
        return xml_str

    def netconf_set_config(self, **kwargs):
        """ netconf_set_config """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = set_nc_config(module, conf_str)
        return xml_str

    def check_bgp_neighbor_af_args(self, **kwargs):
        """ check_bgp_neighbor_af_args """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        vrf_name = module.params['vrf_name']
        if vrf_name:
            if len(vrf_name) > 31 or len(vrf_name) == 0:
                module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
        state = module.params['state']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        if not check_ip_addr(ipaddr=remote_address):
            module.fail_json(msg='Error: The remote_address %s is invalid.' % remote_address)
        conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + CE_GET_BGP_PEER_AF_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if state == 'present':
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<remoteAddress>(.*)</remoteAddress>.*', recv_xml)
                if re_find:
                    result['remote_address'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if remote_address not in re_find:
                        need_cfg = True
                else:
                    need_cfg = True
        elif '<data/>' in recv_xml:
            pass
        else:
            re_find = re.findall('.*<remoteAddress>(.*)</remoteAddress>.*', recv_xml)
            if re_find:
                result['remote_address'] = re_find
                result['vrf_name'] = vrf_name
                result['af_type'] = af_type
                if re_find[0] == remote_address:
                    need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def check_bgp_neighbor_af_other(self, **kwargs):
        """ check_bgp_neighbor_af_other """
        module = kwargs['module']
        result = dict()
        need_cfg = False
        state = module.params['state']
        vrf_name = module.params['vrf_name']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        if state == 'absent':
            result['need_cfg'] = need_cfg
            return result
        advertise_irb = module.params['advertise_irb']
        if advertise_irb != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advertiseIrb></advertiseIrb>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<remoteAddress>%s</remoteAddress>\\s*<advertiseIrb>(.*)</advertiseIrb>.*' % remote_address, recv_xml)
                if re_find:
                    result['advertise_irb'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != advertise_irb:
                        need_cfg = True
                else:
                    need_cfg = True
        advertise_arp = module.params['advertise_arp']
        if advertise_arp != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advertiseArp></advertiseArp>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<remoteAddress>%s</remoteAddress>\\s*.*<advertiseArp>(.*)</advertiseArp>.*' % remote_address, recv_xml)
                if re_find:
                    result['advertise_arp'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != advertise_arp:
                        need_cfg = True
                else:
                    need_cfg = True
        advertise_remote_nexthop = module.params['advertise_remote_nexthop']
        if advertise_remote_nexthop != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advertiseRemoteNexthop></advertiseRemoteNexthop>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<advertiseRemoteNexthop>(.*)</advertiseRemoteNexthop>.*', recv_xml)
                if re_find:
                    result['advertise_remote_nexthop'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != advertise_remote_nexthop:
                        need_cfg = True
                else:
                    need_cfg = True
        advertise_community = module.params['advertise_community']
        if advertise_community != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advertiseCommunity></advertiseCommunity>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<advertiseCommunity>(.*)</advertiseCommunity>.*', recv_xml)
                if re_find:
                    result['advertise_community'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != advertise_community:
                        need_cfg = True
                else:
                    need_cfg = True
        advertise_ext_community = module.params['advertise_ext_community']
        if advertise_ext_community != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advertiseExtCommunity></advertiseExtCommunity>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<advertiseExtCommunity>(.*)</advertiseExtCommunity>.*', recv_xml)
                if re_find:
                    result['advertise_ext_community'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != advertise_ext_community:
                        need_cfg = True
                else:
                    need_cfg = True
        discard_ext_community = module.params['discard_ext_community']
        if discard_ext_community != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<discardExtCommunity></discardExtCommunity>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<discardExtCommunity>(.*)</discardExtCommunity>.*', recv_xml)
                if re_find:
                    result['discard_ext_community'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != discard_ext_community:
                        need_cfg = True
                else:
                    need_cfg = True
        allow_as_loop_enable = module.params['allow_as_loop_enable']
        if allow_as_loop_enable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<allowAsLoopEnable></allowAsLoopEnable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<allowAsLoopEnable>(.*)</allowAsLoopEnable>.*', recv_xml)
                if re_find:
                    result['allow_as_loop_enable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != allow_as_loop_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        allow_as_loop_limit = module.params['allow_as_loop_limit']
        if allow_as_loop_limit:
            if int(allow_as_loop_limit) > 10 or int(allow_as_loop_limit) < 1:
                module.fail_json(msg='the value of allow_as_loop_limit %s is out of [1 - 10].' % allow_as_loop_limit)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<allowAsLoopLimit></allowAsLoopLimit>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<allowAsLoopLimit>(.*)</allowAsLoopLimit>.*', recv_xml)
                if re_find:
                    result['allow_as_loop_limit'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != allow_as_loop_limit:
                        need_cfg = True
                else:
                    need_cfg = True
        keep_all_routes = module.params['keep_all_routes']
        if keep_all_routes != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<keepAllRoutes></keepAllRoutes>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<keepAllRoutes>(.*)</keepAllRoutes>.*', recv_xml)
                if re_find:
                    result['keep_all_routes'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != keep_all_routes:
                        need_cfg = True
                else:
                    need_cfg = True
        nexthop_configure = module.params['nexthop_configure']
        if nexthop_configure:
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<nextHopConfigure></nextHopConfigure>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            self.exist_nexthop_configure = 'null'
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<nextHopConfigure>(.*)</nextHopConfigure>.*', recv_xml)
                if re_find:
                    self.exist_nexthop_configure = re_find[0]
                    result['nexthop_configure'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != nexthop_configure:
                        need_cfg = True
                else:
                    need_cfg = True
        preferred_value = module.params['preferred_value']
        if preferred_value:
            if int(preferred_value) > 65535 or int(preferred_value) < 0:
                module.fail_json(msg='the value of preferred_value %s is out of [0 - 65535].' % preferred_value)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<preferredValue></preferredValue>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<preferredValue>(.*)</preferredValue>.*', recv_xml)
                if re_find:
                    result['preferred_value'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != preferred_value:
                        need_cfg = True
                else:
                    need_cfg = True
        public_as_only = module.params['public_as_only']
        if public_as_only != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<publicAsOnly></publicAsOnly>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<publicAsOnly>(.*)</publicAsOnly>.*', recv_xml)
                if re_find:
                    result['public_as_only'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != public_as_only:
                        need_cfg = True
                else:
                    need_cfg = True
        public_as_only_force = module.params['public_as_only_force']
        if public_as_only_force != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<publicAsOnlyForce></publicAsOnlyForce>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<publicAsOnlyForce>(.*)</publicAsOnlyForce>.*', recv_xml)
                if re_find:
                    result['public_as_only_force'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != public_as_only_force:
                        need_cfg = True
                else:
                    need_cfg = True
        public_as_only_limited = module.params['public_as_only_limited']
        if public_as_only_limited != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<publicAsOnlyLimited></publicAsOnlyLimited>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<publicAsOnlyLimited>(.*)</publicAsOnlyLimited>.*', recv_xml)
                if re_find:
                    result['public_as_only_limited'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != public_as_only_limited:
                        need_cfg = True
                else:
                    need_cfg = True
        public_as_only_replace = module.params['public_as_only_replace']
        if public_as_only_replace != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<publicAsOnlyReplace></publicAsOnlyReplace>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<publicAsOnlyReplace>(.*)</publicAsOnlyReplace>.*', recv_xml)
                if re_find:
                    result['public_as_only_replace'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != public_as_only_replace:
                        need_cfg = True
                else:
                    need_cfg = True
        public_as_only_skip_peer_as = module.params['public_as_only_skip_peer_as']
        if public_as_only_skip_peer_as != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<publicAsOnlySkipPeerAs></publicAsOnlySkipPeerAs>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<publicAsOnlySkipPeerAs>(.*)</publicAsOnlySkipPeerAs>.*', recv_xml)
                if re_find:
                    result['public_as_only_skip_peer_as'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != public_as_only_skip_peer_as:
                        need_cfg = True
                else:
                    need_cfg = True
        route_limit = module.params['route_limit']
        if route_limit:
            if int(route_limit) < 1:
                module.fail_json(msg='the value of route_limit %s is out of [1 - 4294967295].' % route_limit)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<routeLimit></routeLimit>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<routeLimit>(.*)</routeLimit>.*', recv_xml)
                if re_find:
                    result['route_limit'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != route_limit:
                        need_cfg = True
                else:
                    need_cfg = True
        route_limit_percent = module.params['route_limit_percent']
        if route_limit_percent:
            if int(route_limit_percent) < 1 or int(route_limit_percent) > 100:
                module.fail_json(msg='Error: The value of route_limit_percent %s is out of [1 - 100].' % route_limit_percent)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<routeLimitPercent></routeLimitPercent>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<routeLimitPercent>(.*)</routeLimitPercent>.*', recv_xml)
                if re_find:
                    result['route_limit_percent'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != route_limit_percent:
                        need_cfg = True
                else:
                    need_cfg = True
        route_limit_type = module.params['route_limit_type']
        if route_limit_type:
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<routeLimitType></routeLimitType>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<routeLimitType>(.*)</routeLimitType>.*', recv_xml)
                if re_find:
                    result['route_limit_type'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != route_limit_type:
                        need_cfg = True
                else:
                    need_cfg = True
        route_limit_idle_timeout = module.params['route_limit_idle_timeout']
        if route_limit_idle_timeout:
            if int(route_limit_idle_timeout) < 1 or int(route_limit_idle_timeout) > 1200:
                module.fail_json(msg='Error: The value of route_limit_idle_timeout %s is out of [1 - 1200].' % route_limit_idle_timeout)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<routeLimitIdleTimeout></routeLimitIdleTimeout>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<routeLimitIdleTimeout>(.*)</routeLimitIdleTimeout>.*', recv_xml)
                if re_find:
                    result['route_limit_idle_timeout'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != route_limit_idle_timeout:
                        need_cfg = True
                else:
                    need_cfg = True
        rt_updt_interval = module.params['rt_updt_interval']
        if rt_updt_interval:
            if int(rt_updt_interval) < 0 or int(rt_updt_interval) > 600:
                module.fail_json(msg='Error: The value of rt_updt_interval %s is out of [0 - 600].' % rt_updt_interval)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<rtUpdtInterval></rtUpdtInterval>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<rtUpdtInterval>(.*)</rtUpdtInterval>.*', recv_xml)
                if re_find:
                    result['rt_updt_interval'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != rt_updt_interval:
                        need_cfg = True
                else:
                    need_cfg = True
        redirect_ip = module.params['redirect_ip']
        if redirect_ip != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<redirectIP></redirectIP>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<redirectIP>(.*)</redirectIP>.*', recv_xml)
                if re_find:
                    result['redirect_ip'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != redirect_ip:
                        need_cfg = True
                else:
                    need_cfg = True
        redirect_ip_validation = module.params['redirect_ip_validation']
        if redirect_ip_validation != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<redirectIPVaildation></redirectIPVaildation>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<redirectIPVaildation>(.*)</redirectIPVaildation>.*', recv_xml)
                if re_find:
                    result['redirect_ip_validation'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != redirect_ip_validation:
                        need_cfg = True
                else:
                    need_cfg = True
        reflect_client = module.params['reflect_client']
        if reflect_client != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<reflectClient></reflectClient>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<reflectClient>(.*)</reflectClient>.*', recv_xml)
                if re_find:
                    result['reflect_client'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != reflect_client:
                        need_cfg = True
                else:
                    need_cfg = True
        substitute_as_enable = module.params['substitute_as_enable']
        if substitute_as_enable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<substituteAsEnable></substituteAsEnable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<substituteAsEnable>(.*)</substituteAsEnable>.*', recv_xml)
                if re_find:
                    result['substitute_as_enable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != substitute_as_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        import_rt_policy_name = module.params['import_rt_policy_name']
        if import_rt_policy_name:
            if len(import_rt_policy_name) < 1 or len(import_rt_policy_name) > 40:
                module.fail_json(msg='Error: The len of import_rt_policy_name %s is out of [1 - 40].' % import_rt_policy_name)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<importRtPolicyName></importRtPolicyName>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<importRtPolicyName>(.*)</importRtPolicyName>.*', recv_xml)
                if re_find:
                    result['import_rt_policy_name'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != import_rt_policy_name:
                        need_cfg = True
                else:
                    need_cfg = True
        export_rt_policy_name = module.params['export_rt_policy_name']
        if export_rt_policy_name:
            if len(export_rt_policy_name) < 1 or len(export_rt_policy_name) > 40:
                module.fail_json(msg='Error: The len of export_rt_policy_name %s is out of [1 - 40].' % export_rt_policy_name)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<exportRtPolicyName></exportRtPolicyName>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<exportRtPolicyName>(.*)</exportRtPolicyName>.*', recv_xml)
                if re_find:
                    result['export_rt_policy_name'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != export_rt_policy_name:
                        need_cfg = True
                else:
                    need_cfg = True
        import_pref_filt_name = module.params['import_pref_filt_name']
        if import_pref_filt_name:
            if len(import_pref_filt_name) < 1 or len(import_pref_filt_name) > 169:
                module.fail_json(msg='Error: The len of import_pref_filt_name %s is out of [1 - 169].' % import_pref_filt_name)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<importPrefFiltName></importPrefFiltName>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<importPrefFiltName>(.*)</importPrefFiltName>.*', recv_xml)
                if re_find:
                    result['import_pref_filt_name'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != import_pref_filt_name:
                        need_cfg = True
                else:
                    need_cfg = True
        export_pref_filt_name = module.params['export_pref_filt_name']
        if export_pref_filt_name:
            if len(export_pref_filt_name) < 1 or len(export_pref_filt_name) > 169:
                module.fail_json(msg='Error: The len of export_pref_filt_name %s is out of [1 - 169].' % export_pref_filt_name)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<exportPrefFiltName></exportPrefFiltName>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<exportPrefFiltName>(.*)</exportPrefFiltName>.*', recv_xml)
                if re_find:
                    result['export_pref_filt_name'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != export_pref_filt_name:
                        need_cfg = True
                else:
                    need_cfg = True
        import_as_path_filter = module.params['import_as_path_filter']
        if import_as_path_filter:
            if int(import_as_path_filter) < 1 or int(import_as_path_filter) > 256:
                module.fail_json(msg='Error: The value of import_as_path_filter %s is out of [1 - 256].' % import_as_path_filter)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<importAsPathFilter></importAsPathFilter>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<importAsPathFilter>(.*)</importAsPathFilter>.*', recv_xml)
                if re_find:
                    result['import_as_path_filter'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != import_as_path_filter:
                        need_cfg = True
                else:
                    need_cfg = True
        export_as_path_filter = module.params['export_as_path_filter']
        if export_as_path_filter:
            if int(export_as_path_filter) < 1 or int(export_as_path_filter) > 256:
                module.fail_json(msg='Error: The value of export_as_path_filter %s is out of [1 - 256].' % export_as_path_filter)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<exportAsPathFilter></exportAsPathFilter>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<exportAsPathFilter>(.*)</exportAsPathFilter>.*', recv_xml)
                if re_find:
                    result['export_as_path_filter'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != export_as_path_filter:
                        need_cfg = True
                else:
                    need_cfg = True
        import_as_path_name_or_num = module.params['import_as_path_name_or_num']
        if import_as_path_name_or_num:
            if len(import_as_path_name_or_num) < 1 or len(import_as_path_name_or_num) > 51:
                module.fail_json(msg='Error: The len of import_as_path_name_or_num %s is out of [1 - 51].' % import_as_path_name_or_num)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<importAsPathNameOrNum></importAsPathNameOrNum>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<importAsPathNameOrNum>(.*)</importAsPathNameOrNum>.*', recv_xml)
                if re_find:
                    result['import_as_path_name_or_num'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != import_as_path_name_or_num:
                        need_cfg = True
                else:
                    need_cfg = True
        export_as_path_name_or_num = module.params['export_as_path_name_or_num']
        if export_as_path_name_or_num:
            if len(export_as_path_name_or_num) < 1 or len(export_as_path_name_or_num) > 51:
                module.fail_json(msg='Error: The len of export_as_path_name_or_num %s is out of [1 - 51].' % export_as_path_name_or_num)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<exportAsPathNameOrNum></exportAsPathNameOrNum>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<exportAsPathNameOrNum>(.*)</exportAsPathNameOrNum>.*', recv_xml)
                if re_find:
                    result['export_as_path_name_or_num'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != export_as_path_name_or_num:
                        need_cfg = True
                else:
                    need_cfg = True
        import_acl_name_or_num = module.params['import_acl_name_or_num']
        if import_acl_name_or_num:
            if len(import_acl_name_or_num) < 1 or len(import_acl_name_or_num) > 32:
                module.fail_json(msg='Error: The len of import_acl_name_or_num %s is out of [1 - 32].' % import_acl_name_or_num)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<importAclNameOrNum></importAclNameOrNum>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<importAclNameOrNum>(.*)</importAclNameOrNum>.*', recv_xml)
                if re_find:
                    result['import_acl_name_or_num'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != import_acl_name_or_num:
                        need_cfg = True
                else:
                    need_cfg = True
        export_acl_name_or_num = module.params['export_acl_name_or_num']
        if export_acl_name_or_num:
            if len(export_acl_name_or_num) < 1 or len(export_acl_name_or_num) > 32:
                module.fail_json(msg='Error: The len of export_acl_name_or_num %s is out of [1 - 32].' % export_acl_name_or_num)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<exportAclNameOrNum></exportAclNameOrNum>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<exportAclNameOrNum>(.*)</exportAclNameOrNum>.*', recv_xml)
                if re_find:
                    result['export_acl_name_or_num'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != export_acl_name_or_num:
                        need_cfg = True
                else:
                    need_cfg = True
        ipprefix_orf_enable = module.params['ipprefix_orf_enable']
        if ipprefix_orf_enable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<ipprefixOrfEnable></ipprefixOrfEnable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<ipprefixOrfEnable>(.*)</ipprefixOrfEnable>.*', recv_xml)
                if re_find:
                    result['ipprefix_orf_enable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != ipprefix_orf_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        is_nonstd_ipprefix_mod = module.params['is_nonstd_ipprefix_mod']
        if is_nonstd_ipprefix_mod != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<isNonstdIpprefixMod></isNonstdIpprefixMod>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<isNonstdIpprefixMod>(.*)</isNonstdIpprefixMod>.*', recv_xml)
                if re_find:
                    result['is_nonstd_ipprefix_mod'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != is_nonstd_ipprefix_mod:
                        need_cfg = True
                else:
                    need_cfg = True
        orftype = module.params['orftype']
        if orftype:
            if int(orftype) < 0 or int(orftype) > 65535:
                module.fail_json(msg='Error: The value of orftype %s is out of [0 - 65535].' % orftype)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<orftype></orftype>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<orftype>(.*)</orftype>.*', recv_xml)
                if re_find:
                    result['orftype'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != orftype:
                        need_cfg = True
                else:
                    need_cfg = True
        orf_mode = module.params['orf_mode']
        if orf_mode:
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<orfMode></orfMode>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<orfMode>(.*)</orfMode>.*', recv_xml)
                if re_find:
                    result['orf_mode'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != orf_mode:
                        need_cfg = True
                else:
                    need_cfg = True
        soostring = module.params['soostring']
        if soostring:
            if len(soostring) < 3 or len(soostring) > 21:
                module.fail_json(msg='Error: The len of soostring %s is out of [3 - 21].' % soostring)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<soostring></soostring>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<soostring>(.*)</soostring>.*', recv_xml)
                if re_find:
                    result['soostring'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != soostring:
                        need_cfg = True
                else:
                    need_cfg = True
        default_rt_adv_enable = module.params['default_rt_adv_enable']
        if default_rt_adv_enable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<defaultRtAdvEnable></defaultRtAdvEnable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<defaultRtAdvEnable>(.*)</defaultRtAdvEnable>.*', recv_xml)
                if re_find:
                    result['default_rt_adv_enable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != default_rt_adv_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        default_rt_adv_policy = module.params['default_rt_adv_policy']
        if default_rt_adv_policy:
            if len(default_rt_adv_policy) < 1 or len(default_rt_adv_policy) > 40:
                module.fail_json(msg='Error: The len of default_rt_adv_policy %s is out of [1 - 40].' % default_rt_adv_policy)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<defaultRtAdvPolicy></defaultRtAdvPolicy>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<defaultRtAdvPolicy>(.*)</defaultRtAdvPolicy>.*', recv_xml)
                if re_find:
                    result['default_rt_adv_policy'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != default_rt_adv_policy:
                        need_cfg = True
                else:
                    need_cfg = True
        default_rt_match_mode = module.params['default_rt_match_mode']
        if default_rt_match_mode:
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<defaultRtMatchMode></defaultRtMatchMode>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<defaultRtMatchMode>(.*)</defaultRtMatchMode>.*', recv_xml)
                if re_find:
                    result['default_rt_match_mode'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != default_rt_match_mode:
                        need_cfg = True
                else:
                    need_cfg = True
        add_path_mode = module.params['add_path_mode']
        if add_path_mode:
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<addPathMode></addPathMode>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<addPathMode>(.*)</addPathMode>.*', recv_xml)
                if re_find:
                    result['add_path_mode'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != add_path_mode:
                        need_cfg = True
                else:
                    need_cfg = True
        adv_add_path_num = module.params['adv_add_path_num']
        if adv_add_path_num:
            if int(adv_add_path_num) < 2 or int(adv_add_path_num) > 64:
                module.fail_json(msg='Error: The value of adv_add_path_num %s is out of [2 - 64].' % adv_add_path_num)
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<advAddPathNum></advAddPathNum>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<advAddPathNum>(.*)</advAddPathNum>.*', recv_xml)
                if re_find:
                    result['adv_add_path_num'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != adv_add_path_num:
                        need_cfg = True
                else:
                    need_cfg = True
        origin_as_valid = module.params['origin_as_valid']
        if origin_as_valid != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<originAsValid></originAsValid>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<originAsValid>(.*)</originAsValid>.*', recv_xml)
                if re_find:
                    result['origin_as_valid'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != origin_as_valid:
                        need_cfg = True
                else:
                    need_cfg = True
        vpls_enable = module.params['vpls_enable']
        if vpls_enable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<vplsEnable></vplsEnable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<vplsEnable>(.*)</vplsEnable>.*', recv_xml)
                if re_find:
                    result['vpls_enable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != vpls_enable:
                        need_cfg = True
                else:
                    need_cfg = True
        vpls_ad_disable = module.params['vpls_ad_disable']
        if vpls_ad_disable != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<vplsAdDisable></vplsAdDisable>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<vplsAdDisable>(.*)</vplsAdDisable>.*', recv_xml)
                if re_find:
                    result['vpls_ad_disable'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != vpls_ad_disable:
                        need_cfg = True
                else:
                    need_cfg = True
        update_pkt_standard_compatible = module.params['update_pkt_standard_compatible']
        if update_pkt_standard_compatible != 'no_use':
            conf_str = CE_GET_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + '<updatePktStandardCompatible></updatePktStandardCompatible>' + CE_GET_BGP_PEER_AF_TAIL
            recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
            if '<data/>' in recv_xml:
                need_cfg = True
            else:
                re_find = re.findall('.*<updatePktStandardCompatible>(.*)</updatePktStandardCompatible>.*', recv_xml)
                if re_find:
                    result['update_pkt_standard_compatible'] = re_find
                    result['vrf_name'] = vrf_name
                    result['af_type'] = af_type
                    if re_find[0] != update_pkt_standard_compatible:
                        need_cfg = True
                else:
                    need_cfg = True
        result['need_cfg'] = need_cfg
        return result

    def merge_bgp_peer_af(self, **kwargs):
        """ merge_bgp_peer_af """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        conf_str = CE_MERGE_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address) + CE_MERGE_BGP_PEER_AF_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge bgp peer address family failed.')
        cmds = []
        cmd = af_type
        if af_type == 'ipv4uni':
            if vrf_name == '_public_':
                cmd = 'ipv4-family unicast'
            else:
                cmd = 'ipv4-family vpn-instance %s' % vrf_name
        elif af_type == 'ipv4multi':
            cmd = 'ipv4-family multicast'
        elif af_type == 'ipv6uni':
            if vrf_name == '_public_':
                cmd = 'ipv6-family unicast'
            else:
                cmd = 'ipv6-family vpn-instance %s' % vrf_name
        elif af_type == 'evpn':
            cmd = 'l2vpn-family evpn'
        elif af_type == 'ipv4vpn':
            cmd = 'ipv4-family vpnv4'
        elif af_type == 'ipv6vpn':
            cmd = 'ipv6-family vpnv6'
        cmds.append(cmd)
        if vrf_name == '_public_':
            cmd = 'peer %s enable' % remote_address
        else:
            cmd = 'peer %s' % remote_address
        cmds.append(cmd)
        return cmds

    def create_bgp_peer_af(self, **kwargs):
        """ create_bgp_peer_af """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        conf_str = CE_CREATE_BGP_PEER_AF % (vrf_name, af_type, remote_address)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Create bgp peer address family failed.')
        cmds = []
        cmd = af_type
        if af_type == 'ipv4uni':
            if vrf_name == '_public_':
                cmd = 'ipv4-family unicast'
            else:
                cmd = 'ipv4-family vpn-instance %s' % vrf_name
        elif af_type == 'ipv4multi':
            cmd = 'ipv4-family multicast'
        elif af_type == 'ipv6uni':
            if vrf_name == '_public_':
                cmd = 'ipv6-family unicast'
            else:
                cmd = 'ipv6-family vpn-instance %s' % vrf_name
        elif af_type == 'evpn':
            cmd = 'l2vpn-family evpn'
        elif af_type == 'ipv4vpn':
            cmd = 'ipv4-family vpnv4'
        elif af_type == 'ipv6vpn':
            cmd = 'ipv6-family vpnv6'
        cmds.append(cmd)
        if vrf_name == '_public_':
            cmd = 'peer %s enable' % remote_address
        else:
            cmd = 'peer %s' % remote_address
        cmds.append(cmd)
        return cmds

    def delete_bgp_peer_af(self, **kwargs):
        """ delete_bgp_peer_af """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        conf_str = CE_DELETE_BGP_PEER_AF % (vrf_name, af_type, remote_address)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete bgp peer address family failed.')
        cmds = []
        cmd = af_type
        if af_type == 'ipv4uni':
            if vrf_name == '_public_':
                cmd = 'ipv4-family unicast'
            else:
                cmd = 'ipv4-family vpn-instance %s' % vrf_name
        elif af_type == 'ipv4multi':
            cmd = 'ipv4-family multicast'
        elif af_type == 'ipv6uni':
            if vrf_name == '_public_':
                cmd = 'ipv6-family unicast'
            else:
                cmd = 'ipv6-family vpn-instance %s' % vrf_name
        elif af_type == 'evpn':
            cmd = 'l2vpn-family evpn'
        elif af_type == 'ipv4vpn':
            cmd = 'ipv4-family vpnv4'
        elif af_type == 'ipv6vpn':
            cmd = 'ipv6-family vpnv6'
        cmds.append(cmd)
        if vrf_name == '_public_':
            cmd = 'undo peer %s enable' % remote_address
        else:
            cmd = 'undo peer %s' % remote_address
        cmds.append(cmd)
        return cmds

    def merge_bgp_peer_af_other(self, **kwargs):
        """ merge_bgp_peer_af_other """
        module = kwargs['module']
        vrf_name = module.params['vrf_name']
        af_type = module.params['af_type']
        remote_address = module.params['remote_address']
        conf_str = CE_MERGE_BGP_PEER_AF_HEADER % (vrf_name, af_type, remote_address)
        cmds = []
        advertise_irb = module.params['advertise_irb']
        if advertise_irb != 'no_use':
            conf_str += '<advertiseIrb>%s</advertiseIrb>' % advertise_irb
            if advertise_irb == 'true':
                cmd = 'peer %s advertise irb' % remote_address
            else:
                cmd = 'undo peer %s advertise irb' % remote_address
            cmds.append(cmd)
        advertise_arp = module.params['advertise_arp']
        if advertise_arp != 'no_use':
            conf_str += '<advertiseArp>%s</advertiseArp>' % advertise_arp
            if advertise_arp == 'true':
                cmd = 'peer %s advertise arp' % remote_address
            else:
                cmd = 'undo peer %s advertise arp' % remote_address
            cmds.append(cmd)
        advertise_remote_nexthop = module.params['advertise_remote_nexthop']
        if advertise_remote_nexthop != 'no_use':
            conf_str += '<advertiseRemoteNexthop>%s</advertiseRemoteNexthop>' % advertise_remote_nexthop
            if advertise_remote_nexthop == 'true':
                cmd = 'peer %s advertise remote-nexthop' % remote_address
            else:
                cmd = 'undo peer %s advertise remote-nexthop' % remote_address
            cmds.append(cmd)
        advertise_community = module.params['advertise_community']
        if advertise_community != 'no_use':
            conf_str += '<advertiseCommunity>%s</advertiseCommunity>' % advertise_community
            if advertise_community == 'true':
                cmd = 'peer %s advertise-community' % remote_address
            else:
                cmd = 'undo peer %s advertise-community' % remote_address
            cmds.append(cmd)
        advertise_ext_community = module.params['advertise_ext_community']
        if advertise_ext_community != 'no_use':
            conf_str += '<advertiseExtCommunity>%s</advertiseExtCommunity>' % advertise_ext_community
            if advertise_ext_community == 'true':
                cmd = 'peer %s advertise-ext-community' % remote_address
            else:
                cmd = 'undo peer %s advertise-ext-community' % remote_address
            cmds.append(cmd)
        discard_ext_community = module.params['discard_ext_community']
        if discard_ext_community != 'no_use':
            conf_str += '<discardExtCommunity>%s</discardExtCommunity>' % discard_ext_community
            if discard_ext_community == 'true':
                cmd = 'peer %s discard-ext-community' % remote_address
            else:
                cmd = 'undo peer %s discard-ext-community' % remote_address
            cmds.append(cmd)
        allow_as_loop_enable = module.params['allow_as_loop_enable']
        if allow_as_loop_enable != 'no_use':
            conf_str += '<allowAsLoopEnable>%s</allowAsLoopEnable>' % allow_as_loop_enable
            if allow_as_loop_enable == 'true':
                cmd = 'peer %s allow-as-loop' % remote_address
            else:
                cmd = 'undo peer %s allow-as-loop' % remote_address
            cmds.append(cmd)
        allow_as_loop_limit = module.params['allow_as_loop_limit']
        if allow_as_loop_limit:
            conf_str += '<allowAsLoopLimit>%s</allowAsLoopLimit>' % allow_as_loop_limit
            if allow_as_loop_enable == 'true':
                cmd = 'peer %s allow-as-loop %s' % (remote_address, allow_as_loop_limit)
            else:
                cmd = 'undo peer %s allow-as-loop' % remote_address
            cmds.append(cmd)
        keep_all_routes = module.params['keep_all_routes']
        if keep_all_routes != 'no_use':
            conf_str += '<keepAllRoutes>%s</keepAllRoutes>' % keep_all_routes
            if keep_all_routes == 'true':
                cmd = 'peer %s keep-all-routes' % remote_address
            else:
                cmd = 'undo peer %s keep-all-routes' % remote_address
            cmds.append(cmd)
        nexthop_configure = module.params['nexthop_configure']
        if nexthop_configure:
            conf_str += '<nextHopConfigure>%s</nextHopConfigure>' % nexthop_configure
            if nexthop_configure == 'local':
                cmd = 'peer %s next-hop-local' % remote_address
                cmds.append(cmd)
            elif nexthop_configure == 'invariable':
                cmd = 'peer %s next-hop-invariable' % remote_address
                cmds.append(cmd)
            elif self.exist_nexthop_configure != 'null':
                if self.exist_nexthop_configure == 'local':
                    cmd = 'undo peer %s next-hop-local' % remote_address
                    cmds.append(cmd)
                elif self.exist_nexthop_configure == 'invariable':
                    cmd = 'undo peer %s next-hop-invariable' % remote_address
                    cmds.append(cmd)
        preferred_value = module.params['preferred_value']
        if preferred_value:
            conf_str += '<preferredValue>%s</preferredValue>' % preferred_value
            cmd = 'peer %s preferred-value %s' % (remote_address, preferred_value)
            cmds.append(cmd)
        public_as_only = module.params['public_as_only']
        if public_as_only != 'no_use':
            conf_str += '<publicAsOnly>%s</publicAsOnly>' % public_as_only
            if public_as_only == 'true':
                cmd = 'peer %s public-as-only' % remote_address
            else:
                cmd = 'undo peer %s public-as-only' % remote_address
            cmds.append(cmd)
        public_as_only_force = module.params['public_as_only_force']
        if public_as_only_force != 'no_use':
            conf_str += '<publicAsOnlyForce>%s</publicAsOnlyForce>' % public_as_only_force
            if public_as_only_force == 'true':
                cmd = 'peer %s public-as-only force' % remote_address
            else:
                cmd = 'undo peer %s public-as-only force' % remote_address
            cmds.append(cmd)
        public_as_only_limited = module.params['public_as_only_limited']
        if public_as_only_limited != 'no_use':
            conf_str += '<publicAsOnlyLimited>%s</publicAsOnlyLimited>' % public_as_only_limited
            if public_as_only_limited == 'true':
                cmd = 'peer %s public-as-only limited' % remote_address
            else:
                cmd = 'undo peer %s public-as-only limited' % remote_address
            cmds.append(cmd)
        public_as_only_replace = module.params['public_as_only_replace']
        if public_as_only_replace != 'no_use':
            conf_str += '<publicAsOnlyReplace>%s</publicAsOnlyReplace>' % public_as_only_replace
            if public_as_only_replace == 'true':
                if public_as_only_force != 'no_use':
                    cmd = 'peer %s public-as-only force replace' % remote_address
                if public_as_only_limited != 'no_use':
                    cmd = 'peer %s public-as-only limited replace' % remote_address
            else:
                if public_as_only_force != 'no_use':
                    cmd = 'undo peer %s public-as-only force replace' % remote_address
                if public_as_only_limited != 'no_use':
                    cmd = 'undo peer %s public-as-only limited replace' % remote_address
            cmds.append(cmd)
        public_as_only_skip_peer_as = module.params['public_as_only_skip_peer_as']
        if public_as_only_skip_peer_as != 'no_use':
            conf_str += '<publicAsOnlySkipPeerAs>%s</publicAsOnlySkipPeerAs>' % public_as_only_skip_peer_as
            if public_as_only_skip_peer_as == 'true':
                if public_as_only_force != 'no_use':
                    cmd = 'peer %s public-as-only force include-peer-as' % remote_address
                if public_as_only_limited != 'no_use':
                    cmd = 'peer %s public-as-only limited include-peer-as' % remote_address
            else:
                if public_as_only_force != 'no_use':
                    cmd = 'undo peer %s public-as-only force include-peer-as' % remote_address
                if public_as_only_limited != 'no_use':
                    cmd = 'undo peer %s public-as-only limited include-peer-as' % remote_address
            cmds.append(cmd)
        route_limit_sign = 'route-limit'
        if af_type == 'evpn':
            route_limit_sign = 'mac-limit'
        route_limit = module.params['route_limit']
        if route_limit:
            conf_str += '<routeLimit>%s</routeLimit>' % route_limit
            cmd = 'peer %s %s %s' % (remote_address, route_limit_sign, route_limit)
            cmds.append(cmd)
        route_limit_percent = module.params['route_limit_percent']
        if route_limit_percent:
            conf_str += '<routeLimitPercent>%s</routeLimitPercent>' % route_limit_percent
            cmd = 'peer %s %s %s %s' % (remote_address, route_limit_sign, route_limit, route_limit_percent)
            cmds.append(cmd)
        route_limit_type = module.params['route_limit_type']
        if route_limit_type:
            conf_str += '<routeLimitType>%s</routeLimitType>' % route_limit_type
            if route_limit_type == 'alertOnly':
                cmd = 'peer %s %s %s %s alert-only' % (remote_address, route_limit_sign, route_limit, route_limit_percent)
                cmds.append(cmd)
            elif route_limit_type == 'idleForever':
                cmd = 'peer %s %s %s %s idle-forever' % (remote_address, route_limit_sign, route_limit, route_limit_percent)
                cmds.append(cmd)
            elif route_limit_type == 'idleTimeout':
                cmd = 'peer %s %s %s %s idle-timeout' % (remote_address, route_limit_sign, route_limit, route_limit_percent)
                cmds.append(cmd)
        route_limit_idle_timeout = module.params['route_limit_idle_timeout']
        if route_limit_idle_timeout:
            conf_str += '<routeLimitIdleTimeout>%s</routeLimitIdleTimeout>' % route_limit_idle_timeout
            cmd = 'peer %s %s %s %s idle-timeout %s' % (remote_address, route_limit_sign, route_limit, route_limit_percent, route_limit_idle_timeout)
            cmds.append(cmd)
        rt_updt_interval = module.params['rt_updt_interval']
        if rt_updt_interval:
            conf_str += '<rtUpdtInterval>%s</rtUpdtInterval>' % rt_updt_interval
            cmd = 'peer %s route-update-interval %s' % (remote_address, rt_updt_interval)
            cmds.append(cmd)
        redirect_ip = module.params['redirect_ip']
        if redirect_ip != 'no_use':
            conf_str += '<redirectIP>%s</redirectIP>' % redirect_ip
        redirect_ip_validation = module.params['redirect_ip_validation']
        if redirect_ip_validation != 'no_use':
            conf_str += '<redirectIPVaildation>%s</redirectIPVaildation>' % redirect_ip_validation
        reflect_client = module.params['reflect_client']
        if reflect_client != 'no_use':
            conf_str += '<reflectClient>%s</reflectClient>' % reflect_client
            if reflect_client == 'true':
                cmd = 'peer %s reflect-client' % remote_address
            else:
                cmd = 'undo peer %s reflect-client' % remote_address
            cmds.append(cmd)
        substitute_as_enable = module.params['substitute_as_enable']
        if substitute_as_enable != 'no_use':
            conf_str += '<substituteAsEnable>%s</substituteAsEnable>' % substitute_as_enable
            if substitute_as_enable == 'true':
                cmd = 'peer %s substitute-as' % remote_address
            else:
                cmd = 'undo peer %s substitute-as' % remote_address
            cmds.append(cmd)
        import_rt_policy_name = module.params['import_rt_policy_name']
        if import_rt_policy_name:
            conf_str += '<importRtPolicyName>%s</importRtPolicyName>' % import_rt_policy_name
            cmd = 'peer %s route-policy %s import' % (remote_address, import_rt_policy_name)
            cmds.append(cmd)
        export_rt_policy_name = module.params['export_rt_policy_name']
        if export_rt_policy_name:
            conf_str += '<exportRtPolicyName>%s</exportRtPolicyName>' % export_rt_policy_name
            cmd = 'peer %s route-policy %s export' % (remote_address, export_rt_policy_name)
            cmds.append(cmd)
        import_pref_filt_name = module.params['import_pref_filt_name']
        if import_pref_filt_name:
            conf_str += '<importPrefFiltName>%s</importPrefFiltName>' % import_pref_filt_name
            cmd = 'peer %s ip-prefix %s import' % (remote_address, import_pref_filt_name)
            cmds.append(cmd)
        export_pref_filt_name = module.params['export_pref_filt_name']
        if export_pref_filt_name:
            conf_str += '<exportPrefFiltName>%s</exportPrefFiltName>' % export_pref_filt_name
            cmd = 'peer %s ip-prefix %s export' % (remote_address, export_pref_filt_name)
            cmds.append(cmd)
        import_as_path_filter = module.params['import_as_path_filter']
        if import_as_path_filter:
            conf_str += '<importAsPathFilter>%s</importAsPathFilter>' % import_as_path_filter
            cmd = 'peer %s as-path-filter %s import' % (remote_address, import_as_path_filter)
            cmds.append(cmd)
        export_as_path_filter = module.params['export_as_path_filter']
        if export_as_path_filter:
            conf_str += '<exportAsPathFilter>%s</exportAsPathFilter>' % export_as_path_filter
            cmd = 'peer %s as-path-filter %s export' % (remote_address, export_as_path_filter)
            cmds.append(cmd)
        import_as_path_name_or_num = module.params['import_as_path_name_or_num']
        if import_as_path_name_or_num:
            conf_str += '<importAsPathNameOrNum>%s</importAsPathNameOrNum>' % import_as_path_name_or_num
            cmd = 'peer %s as-path-filter %s import' % (remote_address, import_as_path_name_or_num)
            cmds.append(cmd)
        export_as_path_name_or_num = module.params['export_as_path_name_or_num']
        if export_as_path_name_or_num:
            conf_str += '<exportAsPathNameOrNum>%s</exportAsPathNameOrNum>' % export_as_path_name_or_num
            cmd = 'peer %s as-path-filter %s export' % (remote_address, export_as_path_name_or_num)
            cmds.append(cmd)
        import_acl_name_or_num = module.params['import_acl_name_or_num']
        if import_acl_name_or_num:
            conf_str += '<importAclNameOrNum>%s</importAclNameOrNum>' % import_acl_name_or_num
            if import_acl_name_or_num.isdigit():
                cmd = 'peer %s filter-policy %s import' % (remote_address, import_acl_name_or_num)
            else:
                cmd = 'peer %s filter-policy acl-name %s import' % (remote_address, import_acl_name_or_num)
            cmds.append(cmd)
        export_acl_name_or_num = module.params['export_acl_name_or_num']
        if export_acl_name_or_num:
            conf_str += '<exportAclNameOrNum>%s</exportAclNameOrNum>' % export_acl_name_or_num
            if export_acl_name_or_num.isdigit():
                cmd = 'peer %s filter-policy %s export' % (remote_address, export_acl_name_or_num)
            else:
                cmd = 'peer %s filter-policy acl-name %s export' % (remote_address, export_acl_name_or_num)
            cmds.append(cmd)
        ipprefix_orf_enable = module.params['ipprefix_orf_enable']
        if ipprefix_orf_enable != 'no_use':
            conf_str += '<ipprefixOrfEnable>%s</ipprefixOrfEnable>' % ipprefix_orf_enable
            if ipprefix_orf_enable == 'true':
                cmd = 'peer %s capability-advertise orf ip-prefix' % remote_address
            else:
                cmd = 'undo peer %s capability-advertise orf ip-prefix' % remote_address
            cmds.append(cmd)
        is_nonstd_ipprefix_mod = module.params['is_nonstd_ipprefix_mod']
        if is_nonstd_ipprefix_mod != 'no_use':
            conf_str += '<isNonstdIpprefixMod>%s</isNonstdIpprefixMod>' % is_nonstd_ipprefix_mod
            if is_nonstd_ipprefix_mod == 'true':
                if ipprefix_orf_enable == 'true':
                    cmd = 'peer %s capability-advertise orf non-standard-compatible' % remote_address
                else:
                    cmd = 'undo peer %s capability-advertise orf non-standard-compatible' % remote_address
                cmds.append(cmd)
            else:
                if ipprefix_orf_enable == 'true':
                    cmd = 'peer %s capability-advertise orf' % remote_address
                else:
                    cmd = 'undo peer %s capability-advertise orf' % remote_address
                cmds.append(cmd)
        orftype = module.params['orftype']
        if orftype:
            conf_str += '<orftype>%s</orftype>' % orftype
        orf_mode = module.params['orf_mode']
        if orf_mode:
            conf_str += '<orfMode>%s</orfMode>' % orf_mode
            if ipprefix_orf_enable == 'true':
                cmd = 'peer %s capability-advertise orf ip-prefix %s' % (remote_address, orf_mode)
            else:
                cmd = 'undo peer %s capability-advertise orf ip-prefix %s' % (remote_address, orf_mode)
            cmds.append(cmd)
        soostring = module.params['soostring']
        if soostring:
            conf_str += '<soostring>%s</soostring>' % soostring
            cmd = 'peer %s soo %s' % (remote_address, soostring)
            cmds.append(cmd)
        cmd = ''
        default_rt_adv_enable = module.params['default_rt_adv_enable']
        if default_rt_adv_enable != 'no_use':
            conf_str += '<defaultRtAdvEnable>%s</defaultRtAdvEnable>' % default_rt_adv_enable
            if default_rt_adv_enable == 'true':
                cmd += 'peer %s default-route-advertise' % remote_address
            else:
                cmd += 'undo peer %s default-route-advertise' % remote_address
        default_rt_adv_policy = module.params['default_rt_adv_policy']
        if default_rt_adv_policy:
            conf_str += '<defaultRtAdvPolicy>%s</defaultRtAdvPolicy>' % default_rt_adv_policy
            cmd += ' route-policy %s' % default_rt_adv_policy
        default_rt_match_mode = module.params['default_rt_match_mode']
        if default_rt_match_mode:
            conf_str += '<defaultRtMatchMode>%s</defaultRtMatchMode>' % default_rt_match_mode
            if default_rt_match_mode == 'matchall':
                cmd += ' conditional-route-match-all'
            elif default_rt_match_mode == 'matchany':
                cmd += ' conditional-route-match-any'
        if cmd:
            cmds.append(cmd)
        add_path_mode = module.params['add_path_mode']
        if add_path_mode:
            conf_str += '<addPathMode>%s</addPathMode>' % add_path_mode
            if add_path_mode == 'receive':
                cmd = 'peer %s capability-advertise add-path receive' % remote_address
            elif add_path_mode == 'send':
                cmd = 'peer %s capability-advertise add-path send' % remote_address
            elif add_path_mode == 'both':
                cmd = 'peer %s capability-advertise add-path both' % remote_address
            cmds.append(cmd)
        adv_add_path_num = module.params['adv_add_path_num']
        if adv_add_path_num:
            conf_str += '<advAddPathNum>%s</advAddPathNum>' % adv_add_path_num
            cmd = 'peer %s advertise add-path path-number %s' % (remote_address, adv_add_path_num)
            cmds.append(cmd)
        origin_as_valid = module.params['origin_as_valid']
        if origin_as_valid != 'no_use':
            conf_str += '<originAsValid>%s</originAsValid>' % origin_as_valid
        vpls_enable = module.params['vpls_enable']
        if vpls_enable != 'no_use':
            conf_str += '<vplsEnable>%s</vplsEnable>' % vpls_enable
        vpls_ad_disable = module.params['vpls_ad_disable']
        if vpls_ad_disable != 'no_use':
            conf_str += '<vplsAdDisable>%s</vplsAdDisable>' % vpls_ad_disable
        update_pkt_standard_compatible = module.params['update_pkt_standard_compatible']
        if update_pkt_standard_compatible != 'no_use':
            conf_str += '<updatePktStandardCompatible>%s</updatePktStandardCompatible>' % update_pkt_standard_compatible
        conf_str += CE_MERGE_BGP_PEER_AF_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge bgp peer address family other failed.')
        return cmds