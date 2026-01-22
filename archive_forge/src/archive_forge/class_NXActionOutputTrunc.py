import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionOutputTrunc(NXAction):
    """
        Truncate output action

        This action truncate a packet into the specified size and outputs it.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          output(port=port,max_len=max_len)
        ..

        +--------------------------------------------------------------+
        | **output(port**\\=\\ *port*\\,\\ **max_len**\\=\\ *max_len*\\ **)** |
        +--------------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        port             Output port
        max_len          Max bytes to send
        ================ ======================================================

        Example::

            actions += [parser.NXActionOutputTrunc(port=8080,
                                                   max_len=1024)]
        """
    _subtype = nicira_ext.NXAST_OUTPUT_TRUNC
    _fmt_str = '!HI'

    def __init__(self, port, max_len, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionOutputTrunc, self).__init__()
        self.port = port
        self.max_len = max_len

    @classmethod
    def parser(cls, buf):
        port, max_len = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(port, max_len)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.port, self.max_len)
        return data