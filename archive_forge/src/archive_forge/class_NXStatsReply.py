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
@OFPVendorStatsReply.register_stats_vendor(ofproto_common.NX_EXPERIMENTER_ID)
class NXStatsReply(OFPStatsReply):
    _NX_STATS_TYPES = {}

    @staticmethod
    def register_nx_stats_type(body_single_struct=False):

        def _register_nx_stats_type(cls):
            assert cls.cls_stats_type is not None
            assert cls.cls_stats_type not in NXStatsReply._NX_STATS_TYPES
            assert cls.cls_stats_body_cls is not None
            cls.cls_body_single_struct = body_single_struct
            NXStatsReply._NX_STATS_TYPES[cls.cls_stats_type] = cls
            return cls
        return _register_nx_stats_type

    @classmethod
    def parser_stats_body(cls, buf, msg_len, offset):
        body_cls = cls.cls_stats_body_cls
        body = []
        while offset < msg_len:
            entry = body_cls.parser(buf, offset)
            body.append(entry)
            offset += entry.length
        if cls.cls_body_single_struct:
            return body[0]
        return body

    @classmethod
    def parser_stats(cls, datapath, version, msg_type, msg_len, xid, buf, offset):
        msg = MsgBase.parser.__func__(cls, datapath, version, msg_type, msg_len, xid, buf)
        msg.body = msg.parser_stats_body(msg.buf, msg.msg_len, offset)
        return msg

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf, offset):
        type_, = struct.unpack_from(ofproto.NX_STATS_MSG_PACK_STR, bytes(buf), offset)
        offset += ofproto.NX_STATS_MSG0_SIZE
        cls_ = cls._NX_STATS_TYPES.get(type_)
        msg = cls_.parser_stats(datapath, version, msg_type, msg_len, xid, buf, offset)
        return msg