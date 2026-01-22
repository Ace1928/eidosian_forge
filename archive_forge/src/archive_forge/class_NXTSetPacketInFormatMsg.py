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
@_register_exp_type(ofproto_common.NX_EXPERIMENTER_ID, nicira_ext.NXT_SET_PACKET_IN_FORMAT)
class NXTSetPacketInFormatMsg(OFPExperimenter):

    def __init__(self, datapath, packet_in_format):
        super(NXTSetPacketInFormatMsg, self).__init__(datapath, ofproto_common.NX_EXPERIMENTER_ID, nicira_ext.NXT_SET_PACKET_IN_FORMAT)
        self.packet_in_format = packet_in_format

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.experimenter, self.exp_type)
        msg_pack_into(nicira_ext.NX_SET_PACKET_IN_FORMAT_PACK_STR, self.buf, ofproto.OFP_EXPERIMENTER_HEADER_SIZE, self.packet_in_format)

    @classmethod
    def parser_subtype(cls, super_msg):
        packet_in_format = struct.unpack_from(nicira_ext.NX_SET_PACKET_IN_FORMAT_PACK_STR, super_msg.data)[0]
        msg = cls(super_msg.datapath, packet_in_format)
        msg.properties = []
        return msg