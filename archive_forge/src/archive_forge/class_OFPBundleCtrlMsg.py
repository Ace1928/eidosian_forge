import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@_register_parser
@_set_msg_type(ofproto.OFPT_BUNDLE_CONTROL)
class OFPBundleCtrlMsg(MsgBase):
    """
    Bundle control message

    The controller uses this message to create, destroy and commit bundles

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    bundle_id        Id of the bundle
    type             One of the following values.

                     | OFPBCT_OPEN_REQUEST
                     | OFPBCT_OPEN_REPLY
                     | OFPBCT_CLOSE_REQUEST
                     | OFPBCT_CLOSE_REPLY
                     | OFPBCT_COMMIT_REQUEST
                     | OFPBCT_COMMIT_REPLY
                     | OFPBCT_DISCARD_REQUEST
                     | OFPBCT_DISCARD_REPLY
    flags            Bitmap of the following flags.

                     | OFPBF_ATOMIC
                     | OFPBF_ORDERED
    properties       List of ``OFPBundleProp`` subclass instance
    ================ ======================================================

    Example::

        def send_bundle_control(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPBundleCtrlMsg(datapath, 7,
                                              ofp.OFPBCT_OPEN_REQUEST,
                                              ofp.OFPBF_ATOMIC, [])
            datapath.send_msg(req)
    """

    def __init__(self, datapath, bundle_id=None, type_=None, flags=None, properties=None):
        super(OFPBundleCtrlMsg, self).__init__(datapath)
        self.bundle_id = bundle_id
        self.type = type_
        self.flags = flags
        self.properties = properties

    def _serialize_body(self):
        bin_props = bytearray()
        for p in self.properties:
            bin_props += p.serialize()
        msg_pack_into(ofproto.OFP_BUNDLE_CTRL_MSG_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.bundle_id, self.type, self.flags)
        self.buf += bin_props

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPBundleCtrlMsg, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        bundle_id, type_, flags = struct.unpack_from(ofproto.OFP_BUNDLE_CTRL_MSG_PACK_STR, buf, ofproto.OFP_HEADER_SIZE)
        msg.bundle_id = bundle_id
        msg.type = type_
        msg.flags = flags
        msg.properties = []
        rest = msg.buf[ofproto.OFP_BUNDLE_CTRL_MSG_SIZE:]
        while rest:
            p, rest = OFPBundleProp.parse(rest)
            msg.properties.append(p)
        return msg