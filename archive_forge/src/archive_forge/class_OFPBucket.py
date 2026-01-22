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
class OFPBucket(StringifyMixin):

    def __init__(self, weight=0, watch_port=ofproto.OFPP_ANY, watch_group=ofproto.OFPG_ANY, actions=None, len_=None):
        super(OFPBucket, self).__init__()
        self.weight = weight
        self.watch_port = watch_port
        self.watch_group = watch_group
        self.actions = actions

    @classmethod
    def parser(cls, buf, offset):
        len_, weight, watch_port, watch_group = struct.unpack_from(ofproto.OFP_BUCKET_PACK_STR, buf, offset)
        msg = cls(weight, watch_port, watch_group, [])
        msg.len = len_
        length = ofproto.OFP_BUCKET_SIZE
        offset += ofproto.OFP_BUCKET_SIZE
        while length < msg.len:
            action = OFPAction.parser(buf, offset)
            msg.actions.append(action)
            offset += action.len
            length += action.len
        return msg

    def serialize(self, buf, offset):
        action_offset = offset + ofproto.OFP_BUCKET_SIZE
        action_len = 0
        for a in self.actions:
            a.serialize(buf, action_offset)
            action_offset += a.len
            action_len += a.len
        self.len = utils.round_up(ofproto.OFP_BUCKET_SIZE + action_len, 8)
        msg_pack_into(ofproto.OFP_BUCKET_PACK_STR, buf, offset, self.len, self.weight, self.watch_port, self.watch_group)