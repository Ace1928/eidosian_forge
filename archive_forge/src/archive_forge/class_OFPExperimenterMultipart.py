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
class OFPExperimenterMultipart(ofproto_parser.namedtuple('OFPExperimenterMultipart', ('experimenter', 'exp_type', 'data'))):
    """
    The body of OFPExperimenterStatsReply multipart messages.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    experimenter     Experimenter ID
    exp_type         Experimenter defined
    data             Experimenter defined additional data
    ================ ======================================================
    """

    @classmethod
    def parser(cls, buf, offset):
        args = struct.unpack_from(ofproto.OFP_EXPERIMENTER_MULTIPART_HEADER_PACK_STR, buf, offset)
        args = list(args)
        args.append(buf[offset + ofproto.OFP_EXPERIMENTER_MULTIPART_HEADER_SIZE:])
        stats = cls(*args)
        stats.length = ofproto.OFP_METER_FEATURES_SIZE
        return stats

    def serialize(self):
        buf = bytearray()
        msg_pack_into(ofproto.OFP_EXPERIMENTER_MULTIPART_HEADER_PACK_STR, buf, 0, self.experimenter, self.exp_type)
        return buf + self.data