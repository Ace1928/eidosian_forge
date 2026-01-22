import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@_set_stats_type(ofproto.OFPMP_BUNDLE_FEATURES, OFPBundleFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPBundleFeaturesStatsRequest(OFPMultipartRequest):
    """
    Bundle features request message

    The controller uses this message to query a switch about its bundle
    capabilities, including whether it supports atomic bundles, ordered
    bundles, and scheduled bundles.

    ====================== ====================================================
    Attribute              Description
    ====================== ====================================================
    flags                  Zero or ``OFPMPF_REQ_MORE``
    feature_request_flags  Bitmap of the following flags.

                           | OFPBF_TIMESTAMP
                           | OFPBF_TIME_SET_SCHED
    properties             List of ``OFPBundleFeaturesProp`` subclass instance
    ====================== ====================================================

    Example::

        def send_bundle_features_stats_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPBundleFeaturesStatsRequest(datapath, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, feature_request_flags=0, properties=None, type_=None):
        properties = properties if properties else []
        super(OFPBundleFeaturesStatsRequest, self).__init__(datapath, flags)
        self.feature_request_flags = feature_request_flags
        self.properties = properties

    def _serialize_stats_body(self):
        bin_props = bytearray()
        for p in self.properties:
            bin_props += p.serialize()
        msg_pack_into(ofproto.OFP_BUNDLE_FEATURES_REQUEST_PACK_STR, self.buf, ofproto.OFP_MULTIPART_REQUEST_SIZE, self.feature_request_flags)
        self.buf += bin_props