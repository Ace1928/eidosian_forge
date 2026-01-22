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
def _test_set_metadata(self, metadata, mask=None):
    header = ofproto.OXM_OF_METADATA
    match = OFPMatch()
    if mask is None:
        match.set_metadata(metadata)
    else:
        if mask + 1 >> 64 != 1:
            header = ofproto.OXM_OF_METADATA_W
        match.set_metadata_masked(metadata, mask)
        metadata &= mask
    self._test_serialize_and_parser(match, header, metadata, mask)