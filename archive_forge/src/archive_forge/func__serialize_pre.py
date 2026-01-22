import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
def _serialize_pre(self):
    self.version = self.datapath.ofproto.OFP_VERSION
    self.msg_type = self.cls_msg_type
    self.buf = bytearray(self.datapath.ofproto.OFP_HEADER_SIZE)