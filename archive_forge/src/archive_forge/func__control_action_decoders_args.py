import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _control_action_decoders_args():
    return {'resubmit': nested_list_decoder(ListDecoders([('port', decode_default), ('table', decode_int), ('ct', decode_flag)])), 'push': decode_field, 'pop': decode_field, 'exit': decode_flag, 'multipath': nested_list_decoder(ListDecoders([('fields', decode_default), ('basis', decode_int), ('algorithm', decode_default), ('n_links', decode_int), ('arg', decode_int), ('dst', decode_field)]))}