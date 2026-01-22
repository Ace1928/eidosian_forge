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
@OFPAction.register_action_type(ofproto.OFPAT_SET_DL_SRC, ofproto.OFP_ACTION_DL_ADDR_SIZE)
class OFPActionSetDlSrc(OFPActionDlAddr):
    """
    Set the ethernet source address action

    This action indicates the ethernet source address to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    dl_addr          Ethernet address.
    ================ ======================================================
    """

    def __init__(self, dl_addr):
        super(OFPActionSetDlSrc, self).__init__(dl_addr)