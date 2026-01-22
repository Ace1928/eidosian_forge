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
class OFPActionExperimenterUnknown(OFPActionExperimenter):

    def __init__(self, experimenter, data=None, type_=None, len_=None):
        super(OFPActionExperimenterUnknown, self).__init__(experimenter=experimenter)
        self.data = data

    def serialize(self, buf, offset):
        data = self.data
        if data is None:
            data = bytearray()
        self.len = utils.round_up(len(data), 8) + ofproto.OFP_ACTION_EXPERIMENTER_HEADER_SIZE
        super(OFPActionExperimenterUnknown, self).serialize(buf, offset)
        msg_pack_into('!%ds' % len(self.data), buf, offset + ofproto.OFP_ACTION_EXPERIMENTER_HEADER_SIZE, self.data)