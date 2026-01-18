import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def to_match_masked_int(value):
    if isinstance(value, str) and '/' in value:
        value = value.split('/')
        return (str_to_int(value[0]), str_to_int(value[1]))
    return str_to_int(value)