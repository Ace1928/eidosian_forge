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
class OFPHelloElemVersionBitmap(StringifyMixin):
    """
    Version bitmap Hello Element

    ========== =========================================================
    Attribute  Description
    ========== =========================================================
    versions   list of versions of OpenFlow protocol a device supports
    ========== =========================================================
    """

    def __init__(self, versions, type_=None, length=None):
        super(OFPHelloElemVersionBitmap, self).__init__()
        self.type = ofproto.OFPHET_VERSIONBITMAP
        self.length = None
        self._bitmaps = None
        self.versions = versions

    @classmethod
    def parser(cls, buf, offset):
        type_, length = struct.unpack_from(ofproto.OFP_HELLO_ELEM_VERSIONBITMAP_HEADER_PACK_STR, buf, offset)
        assert type_ == ofproto.OFPHET_VERSIONBITMAP
        bitmaps_len = length - ofproto.OFP_HELLO_ELEM_VERSIONBITMAP_HEADER_SIZE
        offset += ofproto.OFP_HELLO_ELEM_VERSIONBITMAP_HEADER_SIZE
        bitmaps = []
        while bitmaps_len >= 4:
            bitmap = struct.unpack_from('!I', buf, offset)
            bitmaps.append(bitmap[0])
            offset += 4
            bitmaps_len -= 4
        versions = [i * 32 + shift for i, bitmap in enumerate(bitmaps) for shift in range(31) if bitmap & 1 << shift]
        elem = cls(versions)
        elem.length = length
        elem._bitmaps = bitmaps
        return elem