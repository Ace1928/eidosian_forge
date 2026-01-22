import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
class NXTSetAsyncConfig(NiciraHeader):

    def __init__(self, datapath, packet_in_mask, port_status_mask, flow_removed_mask):
        super(NXTSetAsyncConfig, self).__init__(datapath, ofproto.NXT_SET_ASYNC_CONFIG)
        self.packet_in_mask = packet_in_mask
        self.port_status_mask = port_status_mask
        self.flow_removed_mask = flow_removed_mask

    def _serialize_body(self):
        self.serialize_header()
        msg_pack_into(ofproto.NX_ASYNC_CONFIG_PACK_STR, self.buf, ofproto.NICIRA_HEADER_SIZE, self.packet_in_mask[0], self.packet_in_mask[1], self.port_status_mask[0], self.port_status_mask[1], self.flow_removed_mask[0], self.flow_removed_mask[1])