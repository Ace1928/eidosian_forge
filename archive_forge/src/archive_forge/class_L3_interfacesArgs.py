from __future__ import absolute_import, division, print_function
class L3_interfacesArgs(object):
    """The arg spec for the vyos_l3_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'ipv4': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'ipv6': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'name': {'required': True, 'type': 'str'}, 'vifs': {'elements': 'dict', 'options': {'ipv4': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'ipv6': {'elements': 'dict', 'options': {'address': {'type': 'str'}}, 'type': 'list'}, 'vlan_id': {'type': 'int'}}, 'type': 'list'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'gathered', 'parsed'], 'default': 'merged', 'type': 'str'}}