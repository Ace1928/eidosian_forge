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
@_set_msg_type(ofproto.OFPT_ROLE_REQUEST)
class OFPRoleRequest(MsgBase):
    """
    Role request message

    The controller uses this message to change its role.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    role             One of the following values.

                     | OFPCR_ROLE_NOCHANGE
                     | OFPCR_ROLE_EQUAL
                     | OFPCR_ROLE_MASTER
                     | OFPCR_ROLE_SLAVE
    generation_id    Master Election Generation ID
    ================ ======================================================

    Example::

        def send_role_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPRoleRequest(datapath, ofp.OFPCR_ROLE_EQUAL, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, role=None, generation_id=None):
        super(OFPRoleRequest, self).__init__(datapath)
        self.role = role
        self.generation_id = generation_id

    def _serialize_body(self):
        assert self.role is not None
        assert self.generation_id is not None
        msg_pack_into(ofproto.OFP_ROLE_REQUEST_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.role, self.generation_id)