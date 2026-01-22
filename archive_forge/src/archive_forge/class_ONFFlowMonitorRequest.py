import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class ONFFlowMonitorRequest(StringifyMixin):

    def __init__(self, id_, flags, match=OFPMatch(), out_port=ofproto.OFPP_ANY, table_id=ofproto.OFPTT_ALL, match_len=None):
        self.id = id_
        self.flags = flags
        self.match_len = match_len
        self.out_port = out_port
        self.table_id = table_id
        self.match = match

    def serialize(self):
        match = self.match
        bin_match = bytearray()
        ofp_match_len = match.serialize(bin_match, 0)
        assert len(bin_match) == ofp_match_len
        match_len = match.length
        match_hdr_len = ofproto.OFP_MATCH_SIZE - 4
        bin_match = bytearray(bin_match)[match_hdr_len:match_len]
        self.match_len = len(bin_match)
        buf = bytearray()
        msg_pack_into(ofproto.ONF_FLOW_MONITOR_REQUEST_PACK_STR, buf, 0, self.id, self.flags, self.match_len, self.out_port, self.table_id)
        buf += bin_match
        pad_len = utils.round_up(self.match_len, 8) - self.match_len
        buf += pad_len * b'\x00'
        return buf