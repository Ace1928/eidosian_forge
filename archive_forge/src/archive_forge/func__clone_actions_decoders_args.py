import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _clone_actions_decoders_args(action_decoders):
    """Generate the decoder arguments for the clone actions.

        Args:
            action_decoders (dict): The decoders of the supported nested
            actions.
        """
    return {'learn': decode_learn(action_decoders), 'clone': nested_kv_decoder(KVDecoders(action_decoders, default_free=decode_free_output, ignore_case=True), is_list=True), 'write_actions': nested_kv_decoder(KVDecoders(action_decoders, default_free=decode_free_output, ignore_case=True), is_list=True)}