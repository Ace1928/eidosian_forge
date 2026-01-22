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
@_set_stats_type(ofproto.OFPMP_TABLE_DESC, OFPTableDesc)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPTableDescStatsRequest(OFPMultipartRequest):
    """
    Table description request message

    The controller uses this message to query description of all the tables.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    ================ ======================================================

    Example::

        def send_table_desc_stats_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPTableDescStatsRequest(datapath, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, type_=None):
        super(OFPTableDescStatsRequest, self).__init__(datapath, flags)