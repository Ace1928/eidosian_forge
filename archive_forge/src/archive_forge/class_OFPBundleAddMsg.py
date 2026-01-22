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
@_set_msg_type(ofproto.OFPT_BUNDLE_ADD_MESSAGE)
class OFPBundleAddMsg(MsgInMsgBase):
    """
    Bundle add message

    The controller uses this message to add a message to a bundle

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    bundle_id        Id of the bundle
    flags            Bitmap of the following flags.

                     | OFPBF_ATOMIC
                     | OFPBF_ORDERED
    message          ``MsgBase`` subclass instance
    properties       List of ``OFPBundleProp`` subclass instance
    ================ ======================================================

    Example::

        def send_bundle_add_message(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            msg = ofp_parser.OFPRoleRequest(datapath, ofp.OFPCR_ROLE_EQUAL, 0)

            req = ofp_parser.OFPBundleAddMsg(datapath, 7, ofp.OFPBF_ATOMIC,
                                             msg, [])
            datapath.send_msg(req)
    """

    def __init__(self, datapath, bundle_id, flags, message, properties):
        super(OFPBundleAddMsg, self).__init__(datapath)
        self.bundle_id = bundle_id
        self.flags = flags
        self.message = message
        self.properties = properties

    def _serialize_body(self):
        if self.message.xid != self.xid:
            self.message.set_xid(self.xid)
        self.message.serialize()
        tail_buf = self.message.buf
        if len(self.properties) > 0:
            message_len = len(tail_buf)
            pad_len = utils.round_up(message_len, 8) - message_len
            msg_pack_into('%dx' % pad_len, tail_buf, message_len)
        for p in self.properties:
            tail_buf += p.serialize()
        msg_pack_into(ofproto.OFP_BUNDLE_ADD_MSG_0_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.bundle_id, self.flags)
        self.buf += tail_buf