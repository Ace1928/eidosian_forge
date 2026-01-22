from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@LSA.register_type(OSPF_OPAQUE_LINK_LSA)
class LocalOpaqueLSA(OpaqueLSA):

    def __init__(self, ls_age=0, options=0, type_=OSPF_OPAQUE_LINK_LSA, adv_router='0.0.0.0', ls_seqnum=0, checksum=0, length=0, opaque_type=OSPF_OPAQUE_TYPE_UNKNOWN, opaque_id=0, data=None):
        self.data = data
        super(LocalOpaqueLSA, self).__init__(ls_age, options, type_, 0, adv_router, ls_seqnum, checksum, length, opaque_type, opaque_id)