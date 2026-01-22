import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _make_exp_hdr(oxx, mod, n):
    exp_hdr = bytearray()
    try:
        get_desc = getattr(mod, '_' + oxx + '_field_desc')
        desc = get_desc(n)
    except KeyError:
        return (n, exp_hdr)
    if desc._class == OFPXXC_EXPERIMENTER:
        exp_id, exp_type = n
        assert desc.experimenter_id == exp_id
        oxx_type = getattr(desc, oxx + '_type')
        if desc.exp_type == 2560:
            exp_hdr_pack_str = '!IH'
            msg_pack_into(exp_hdr_pack_str, exp_hdr, 0, desc.experimenter_id, desc.exp_type)
        else:
            assert oxx_type == exp_type | OFPXXC_EXPERIMENTER << 7
            exp_hdr_pack_str = '!I'
            msg_pack_into(exp_hdr_pack_str, exp_hdr, 0, desc.experimenter_id)
        assert len(exp_hdr) == struct.calcsize(exp_hdr_pack_str)
        n = oxx_type
        assert n >> 7 == OFPXXC_EXPERIMENTER
    return (n, exp_hdr)