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
@_register_parser
@_set_msg_type(ofproto.OFPT_VENDOR)
class OFPVendor(MsgBase):
    """
    Vendor message

    The controller send this message to send the vendor-specific
    information to a switch.
    """
    _VENDORS = {}

    @staticmethod
    def register_vendor(id_):

        def _register_vendor(cls):
            OFPVendor._VENDORS[id_] = cls
            return cls
        return _register_vendor

    def __init__(self, datapath):
        super(OFPVendor, self).__init__(datapath)
        self.data = None
        self.vendor = None

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPVendor, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.vendor, = struct.unpack_from(ofproto.OFP_VENDOR_HEADER_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        cls_ = cls._VENDORS.get(msg.vendor)
        if cls_:
            msg.data = cls_.parser(datapath, msg.buf, 0)
        else:
            msg.data = msg.buf[ofproto.OFP_VENDOR_HEADER_SIZE:]
        return msg

    def serialize_header(self):
        msg_pack_into(ofproto.OFP_VENDOR_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.vendor)

    def _serialize_body(self):
        assert self.data is not None
        self.serialize_header()
        self.buf += self.data