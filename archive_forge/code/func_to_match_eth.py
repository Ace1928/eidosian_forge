import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def to_match_eth(value):
    if '/' in value:
        value = value.split('/')
        return (value[0], value[1])
    return value