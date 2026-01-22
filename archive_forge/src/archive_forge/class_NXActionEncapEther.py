import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionEncapEther(NXAction):
    """
        Encap Ether

        This action encaps package with ethernet

        And equivalent to the followings action of ovs-ofctl command.

        ::

            encap(ethernet)

        Example::

            actions += [parser.NXActionEncapEther()]
        """
    _subtype = nicira_ext.NXAST_RAW_ENCAP
    _fmt_str = '!HI'

    def __init__(self, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionEncapEther, self).__init__()
        self.hdr_size = 0
        self.new_pkt_type = 0

    @classmethod
    def parser(cls, buf):
        return cls()

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.hdr_size, self.new_pkt_type)
        return data