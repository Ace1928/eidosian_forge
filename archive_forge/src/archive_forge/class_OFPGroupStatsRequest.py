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
@_set_stats_type(ofproto.OFPMP_GROUP, OFPGroupStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPGroupStatsRequest(OFPMultipartRequest):
    """
    Group statistics request message

    The controller uses this message to query statistics of one or more
    groups.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    group_id         ID of group to read (OFPG_ALL to all groups)
    ================ ======================================================

    Example::

        def send_group_stats_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPGroupStatsRequest(datapath, 0, ofp.OFPG_ALL)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, group_id=ofproto.OFPG_ALL, type_=None):
        super(OFPGroupStatsRequest, self).__init__(datapath, flags)
        self.group_id = group_id

    def _serialize_stats_body(self):
        msg_pack_into(ofproto.OFP_GROUP_STATS_REQUEST_PACK_STR, self.buf, ofproto.OFP_MULTIPART_REQUEST_SIZE, self.group_id)