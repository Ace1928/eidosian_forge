import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _field_action_decoders_args():
    """Returns the decoders arguments for field-modification actions."""
    field_default_decoders = ['set_mpls_label', 'set_mpls_tc', 'set_mpls_ttl', 'mod_nw_tos', 'mod_nw_ecn', 'mod_tp_src', 'mod_tp_dst']
    return {'load': decode_load_field, 'set_field': functools.partial(decode_set_field, KVDecoders(OFPFlow._field_decoder_args())), 'move': decode_move_field, 'mod_dl_dst': EthMask, 'mod_dl_src': EthMask, 'mod_nw_dst': IPMask, 'mod_nw_src': IPMask, 'mod_nw_ttl': decode_int, 'mod_vlan_vid': decode_int, 'set_vlan_vid': decode_int, 'mod_vlan_pcp': decode_int, 'set_vlan_pcp': decode_int, 'dec_ttl': decode_dec_ttl, 'dec_mpls_ttl': decode_flag, 'dec_nsh_ttl': decode_flag, 'delete_field': decode_field, 'check_pkt_larger': decode_chk_pkt_larger, **{field: decode_default for field in field_default_decoders}}