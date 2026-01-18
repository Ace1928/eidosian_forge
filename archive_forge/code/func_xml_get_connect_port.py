from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def xml_get_connect_port(self):
    """ Get connect port by xml """
    tmp_cfg = None
    conf_str = CE_GET_SNMP_PORT
    recv_xml = self.netconf_get_config(conf_str=conf_str)
    if '<data/>' in recv_xml:
        pass
    else:
        xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        snmp_port_info = root.findall('snmp/systemCfg/snmpListenPort')
        if snmp_port_info:
            tmp_cfg = snmp_port_info[0].text
        return tmp_cfg