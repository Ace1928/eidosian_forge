import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _calc_len(self, len_):
    for tlv_ in self.tlvs:
        len_ += len(tlv_)
    len_ += self._END_TLV_LEN
    return len_