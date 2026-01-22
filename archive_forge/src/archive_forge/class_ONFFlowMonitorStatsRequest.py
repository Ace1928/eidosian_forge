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
@_set_stats_type(ofproto.OFPMP_EXPERIMENTER, OFPExperimenterMultipart)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class ONFFlowMonitorStatsRequest(OFPExperimenterStatsRequestBase):
    """
    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    body             List of ONFFlowMonitorRequest instances
    ================ ======================================================
    """

    def __init__(self, datapath, flags, body=None, type_=None, experimenter=None, exp_type=None):
        body = body if body else []
        super(ONFFlowMonitorStatsRequest, self).__init__(datapath, flags, experimenter=ofproto_common.ONF_EXPERIMENTER_ID, exp_type=ofproto.ONFMP_FLOW_MONITOR)
        self.body = body

    def _serialize_stats_body(self):
        data = bytearray()
        for i in self.body:
            data += i.serialize()
        body = OFPExperimenterMultipart(experimenter=self.experimenter, exp_type=self.exp_type, data=data)
        self.buf += body.serialize()