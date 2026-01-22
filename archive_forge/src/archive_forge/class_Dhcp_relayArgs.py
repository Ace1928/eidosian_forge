from __future__ import absolute_import, division, print_function
class Dhcp_relayArgs(object):
    """The arg spec for the sonic_dhcp_relay module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'ipv4': {'options': {'circuit_id': {'choices': ['%h:%p', '%i', '%p'], 'type': 'str'}, 'link_select': {'type': 'bool'}, 'max_hop_count': {'type': 'int'}, 'policy_action': {'choices': ['append', 'discard', 'replace'], 'type': 'str'}, 'server_addresses': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'source_interface': {'type': 'str'}, 'vrf_name': {'type': 'str'}, 'vrf_select': {'type': 'bool'}}, 'type': 'dict'}, 'ipv6': {'options': {'max_hop_count': {'type': 'int'}, 'server_addresses': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'source_interface': {'type': 'str'}, 'vrf_name': {'type': 'str'}, 'vrf_select': {'type': 'bool'}}, 'type': 'dict'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}