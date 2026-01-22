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
@OFPAction.register_action_type(ofproto.OFPAT_SET_MPLS_TTL, ofproto.OFP_ACTION_MPLS_TTL_SIZE)
class OFPActionSetMplsTtl(OFPAction):
    """
    Set MPLS TTL action

    This action sets the MPLS TTL.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    mpls_ttl         MPLS TTL
    ================ ======================================================
    """

    def __init__(self, mpls_ttl, type_=None, len_=None):
        super(OFPActionSetMplsTtl, self).__init__()
        self.mpls_ttl = mpls_ttl

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, mpls_ttl = struct.unpack_from(ofproto.OFP_ACTION_MPLS_TTL_PACK_STR, buf, offset)
        return cls(mpls_ttl)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_MPLS_TTL_PACK_STR, buf, offset, self.type, self.len, self.mpls_ttl)