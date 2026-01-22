from __future__ import absolute_import, division, print_function
class Routing_optionsArgs(object):
    """The arg spec for the junos_routing_options module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'autonomous_system': {'options': {'as_number': {'type': 'str', 'required': True}, 'asdot_notation': {'type': 'bool'}, 'loops': {'type': 'int'}}, 'type': 'dict'}, 'router_id': {'type': 'str'}}, 'type': 'dict'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'deleted', 'overridden', 'parsed', 'gathered', 'rendered'], 'default': 'merged', 'type': 'str'}}