import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionDecTtl(NXAction):
    """
        Decrement IP TTL action

        This action decrements TTL of IPv4 packet or
        hop limit of IPv6 packet.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          dec_ttl
        ..

        +-------------+
        | **dec_ttl** |
        +-------------+

        .. NOTE::
            This actions is supported by
            ``OFPActionDecNwTtl``
            in OpenFlow1.2 or later.

        Example::

            actions += [parser.NXActionDecTtl()]
        """
    _subtype = nicira_ext.NXAST_DEC_TTL
    _fmt_str = '!6x'

    def __init__(self, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionDecTtl, self).__init__()

    @classmethod
    def parser(cls, buf):
        return cls()

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0)
        return data