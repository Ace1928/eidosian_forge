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
@_register_exp_type(ofproto_common.NX_EXPERIMENTER_ID, nicira_ext.NXT_PACKET_IN2)
class NXTPacketIn2(OFPExperimenter):

    def __init__(self, datapath, properties=None):
        super(NXTPacketIn2, self).__init__(datapath, ofproto_common.NX_EXPERIMENTER_ID, nicira_ext.NXT_PACKET_IN2)
        self.properties = properties or []
        self.data = None
        self.total_len = None
        self.buffer_id = None
        self.table_id = None
        self.cookie = None
        self.reason = None
        self.metadata = None
        self.userdata = None
        self.continuation = None

    def _serialize_body(self):
        bin_props = bytearray()
        for p in self.properties:
            bin_props += p.serialize()
        msg_pack_into(ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.experimenter, self.exp_type)
        self.buf += bin_props

    @classmethod
    def parser_subtype(cls, super_msg):
        msg = cls(super_msg.datapath)
        rest = super_msg.data
        while rest:
            p, rest = NXTPacketIn2Prop.parse(rest)
            msg.properties.append(p)
            msg._parse_property(p)
        return msg

    def _parse_property(self, p):
        if p.type == nicira_ext.NXPINT_PACKET:
            self.data = p.data
            self.total_len = p.length
        elif p.type == nicira_ext.NXPINT_FULL_LEN:
            self.total_len = struct.unpack_from('!I', p.data)[0]
        elif p.type == nicira_ext.NXPINT_BUFFER_ID:
            self.buffer_id = struct.unpack_from('!I', p.data)[0]
        elif p.type == nicira_ext.NXPINT_TABLE_ID:
            self.table_id = struct.unpack_from('B', p.data)[0]
        elif p.type == nicira_ext.NXPINT_COOKIE:
            self.cookie = struct.unpack_from('!Q', p.data)[0]
        elif p.type == nicira_ext.NXPINT_REASON:
            self.reason = struct.unpack_from('!B', p.data)[0]
        elif p.type == nicira_ext.NXPINT_METADATA:
            self.metadata = p.data
        elif p.type == nicira_ext.NXPINT_USERDATA:
            self.userdata = p.data
        elif p.type == nicira_ext.NXPINT_CONTINUATION:
            self.continuation = p.data