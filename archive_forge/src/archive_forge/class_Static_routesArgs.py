from __future__ import absolute_import, division, print_function
class Static_routesArgs(object):
    """The arg spec for the vyos_static_routes module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'address_families': {'elements': 'dict', 'options': {'afi': {'choices': ['ipv4', 'ipv6'], 'required': True, 'type': 'str'}, 'routes': {'elements': 'dict', 'options': {'blackhole_config': {'options': {'distance': {'type': 'int'}, 'type': {'type': 'str'}}, 'type': 'dict'}, 'dest': {'required': True, 'type': 'str'}, 'next_hops': {'elements': 'dict', 'options': {'admin_distance': {'type': 'int'}, 'enabled': {'type': 'bool'}, 'forward_router_address': {'required': True, 'type': 'str'}, 'interface': {'type': 'str'}}, 'type': 'list'}}, 'type': 'list'}}, 'type': 'list'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}