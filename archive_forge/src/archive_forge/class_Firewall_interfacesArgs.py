from __future__ import absolute_import, division, print_function
class Firewall_interfacesArgs(object):
    """The arg spec for the vyos_firewall_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'access_rules': {'elements': 'dict', 'options': {'afi': {'choices': ['ipv4', 'ipv6'], 'required': True, 'type': 'str'}, 'rules': {'elements': 'dict', 'options': {'direction': {'choices': ['in', 'local', 'out'], 'required': True, 'type': 'str'}, 'name': {'type': 'str'}}, 'type': 'list'}}, 'type': 'list'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'parsed', 'rendered', 'gathered'], 'default': 'merged', 'type': 'str'}}