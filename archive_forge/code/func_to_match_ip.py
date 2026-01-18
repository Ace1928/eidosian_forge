import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def to_match_ip(value):
    if '/' in value:
        ip_addr, ip_mask = value.split('/')
        if ip_mask.isdigit():
            ip = netaddr.ip.IPNetwork(value)
            ip_addr = str(ip.ip)
            ip_mask = str(ip.netmask)
        return (ip_addr, ip_mask)
    return value