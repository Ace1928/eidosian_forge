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
class OFPPacketQueue(StringifyMixin):

    def __init__(self, queue_id, port, properties, len_=None):
        super(OFPPacketQueue, self).__init__()
        self.queue_id = queue_id
        self.port = port
        self.len = len_
        self.properties = properties

    @classmethod
    def parser(cls, buf, offset):
        queue_id, port, len_ = struct.unpack_from(ofproto.OFP_PACKET_QUEUE_PACK_STR, buf, offset)
        length = ofproto.OFP_PACKET_QUEUE_SIZE
        offset += ofproto.OFP_PACKET_QUEUE_SIZE
        properties = []
        while length < len_:
            queue_prop = OFPQueueProp.parser(buf, offset)
            if queue_prop is not None:
                properties.append(queue_prop)
                offset += queue_prop.len
                length += queue_prop.len
        o = cls(queue_id, port, properties)
        o.len = len_
        return o