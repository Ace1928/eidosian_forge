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
@_set_msg_type(ofproto.OFPT_METER_MOD)
class OFPMeterMod(MsgBase):
    """
    Meter modification message

    The controller sends this message to modify the meter.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    command          One of the following values.

                     | OFPMC_ADD
                     | OFPMC_MODIFY
                     | OFPMC_DELETE
    flags            Bitmap of the following flags.

                     | OFPMF_KBPS
                     | OFPMF_PKTPS
                     | OFPMF_BURST
                     | OFPMF_STATS
    meter_id         Meter instance
    bands            list of the following class instance.

                     | OFPMeterBandDrop
                     | OFPMeterBandDscpRemark
                     | OFPMeterBandExperimenter
    ================ ======================================================
    """

    def __init__(self, datapath, command=ofproto.OFPMC_ADD, flags=ofproto.OFPMF_KBPS, meter_id=1, bands=None):
        bands = bands if bands else []
        super(OFPMeterMod, self).__init__(datapath)
        self.command = command
        self.flags = flags
        self.meter_id = meter_id
        self.bands = bands

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_METER_MOD_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.command, self.flags, self.meter_id)
        offset = ofproto.OFP_METER_MOD_SIZE
        for b in self.bands:
            b.serialize(self.buf, offset)
            offset += b.len