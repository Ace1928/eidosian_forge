import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@staticmethod
def _parse_subfield(buf):
    n, len = ofp.oxm_parse_header(buf, 0)
    assert len == 4
    field = ofp.oxm_to_user_header(n)
    rest = buf[len:]
    ofs, = struct.unpack_from('!H', rest, 0)
    return (field, ofs)