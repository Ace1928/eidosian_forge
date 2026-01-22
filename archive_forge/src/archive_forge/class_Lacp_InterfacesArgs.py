from __future__ import absolute_import, division, print_function
class Lacp_InterfacesArgs(object):
    """The arg spec for the ios_lacp_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'name': {'required': True, 'type': 'str'}, 'port_priority': {'type': 'int'}, 'fast_switchover': {'type': 'bool'}, 'max_bundle': {'type': 'int'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'parsed', 'gathered'], 'default': 'merged', 'type': 'str'}}