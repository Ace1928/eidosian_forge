from __future__ import absolute_import, division, print_function
class Vlan_mappingArgs(object):
    """The arg spec for the sonic_vlan_mapping module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'mapping': {'elements': 'dict', 'options': {'dot1q_tunnel': {'type': 'bool', 'default': False}, 'inner_vlan': {'type': 'int'}, 'priority': {'type': 'int'}, 'service_vlan': {'required': True, 'type': 'int'}, 'vlan_ids': {'elements': 'str', 'type': 'list'}}, 'type': 'list'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}