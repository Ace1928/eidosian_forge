import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _extra_match_decoder_args():
    """Returns the extra KVDecoder arguments needed to decode the match
        part of a flow (apart from the fields)."""
    return {'priority': decode_int}