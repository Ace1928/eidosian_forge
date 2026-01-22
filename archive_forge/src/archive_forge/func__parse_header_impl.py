import struct
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib import type_desc
def _parse_header_impl(mod, buf, offset):
    hdr_pack_str = '!I'
    header, = struct.unpack_from(hdr_pack_str, buf, offset)
    hdr_len = struct.calcsize(hdr_pack_str)
    oxx_type = header >> 9
    oxm_hasmask = mod.oxm_tlv_header_extract_hasmask(header)
    oxx_class = oxx_type >> 7
    oxx_length = header & 255
    if oxx_class == OFPXXC_EXPERIMENTER:
        exp_hdr_pack_str = '!I'
        exp_id, = struct.unpack_from(exp_hdr_pack_str, buf, offset + hdr_len)
        exp_hdr_len = struct.calcsize(exp_hdr_pack_str)
        assert exp_hdr_len == 4
        oxx_field = oxx_type & 127
        if exp_id == ofproto_common.ONF_EXPERIMENTER_ID and oxx_field == 0:
            onf_exp_type_pack_str = '!H'
            exp_type, = struct.unpack_from(onf_exp_type_pack_str, buf, offset + hdr_len + exp_hdr_len)
            exp_hdr_len += struct.calcsize(onf_exp_type_pack_str)
            assert exp_hdr_len == 4 + 2
            num = (exp_id, exp_type)
        else:
            num = (exp_id, oxx_type)
    else:
        num = oxx_type
        exp_hdr_len = 0
    value_len = oxx_length - exp_hdr_len
    if oxm_hasmask:
        value_len //= 2
    assert value_len > 0
    field_len = hdr_len + oxx_length
    total_hdr_len = hdr_len + exp_hdr_len
    return (num, total_hdr_len, oxm_hasmask, value_len, field_len)