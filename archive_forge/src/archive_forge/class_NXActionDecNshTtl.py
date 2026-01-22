import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionDecNshTtl(NXAction):
    """
        Decrement NSH TTL action

        This action decrements the TTL in the Network Service Header(NSH).

        This action was added in OVS v2.9.

        And equivalent to the followings action of ovs-ofctl command.

        ::

            dec_nsh_ttl

        Example::

            actions += [parser.NXActionDecNshTtl()]
        """
    _subtype = nicira_ext.NXAST_DEC_NSH_TTL
    _fmt_str = '!6x'

    def __init__(self, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionDecNshTtl, self).__init__()

    @classmethod
    def parser(cls, buf):
        return cls()

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0)
        return data