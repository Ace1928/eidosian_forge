import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
def _test_set_arp_tha(self, arp_tha, mask=None):
    header = ofproto.OXM_OF_ARP_THA
    match = OFPMatch()
    arp_tha = mac.haddr_to_bin(arp_tha)
    if mask is None:
        match.set_arp_tha(arp_tha)
    else:
        header = ofproto.OXM_OF_ARP_THA_W
        mask = mac.haddr_to_bin(mask)
        match.set_arp_tha_masked(arp_tha, mask)
        arp_tha = mac.haddr_bitand(arp_tha, mask)
    self._test_serialize_and_parser(match, header, arp_tha, mask)