from __future__ import absolute_import, division, print_function
class Dhcp_snoopingArgs(object):
    """The arg spec for the sonic_dhcp_snooping module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'afis': {'elements': 'dict', 'options': {'afi': {'choices': ['ipv4', 'ipv6'], 'required': True, 'type': 'str'}, 'enabled': {'type': 'bool'}, 'source_bindings': {'elements': 'dict', 'options': {'mac_addr': {'required': True, 'type': 'str'}, 'ip_addr': {'type': 'str'}, 'intf_name': {'type': 'str'}, 'vlan_id': {'type': 'int'}}, 'type': 'list'}, 'trusted': {'elements': 'dict', 'options': {'intf_name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'verify_mac': {'type': 'bool'}, 'vlans': {'elements': 'str', 'type': 'list'}}, 'type': 'list'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'deleted', 'overridden', 'replaced'], 'default': 'merged', 'type': 'str'}}