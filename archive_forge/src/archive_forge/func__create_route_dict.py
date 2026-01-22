from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.static_routes.static_routes import (
def _create_route_dict(self, afi, route_path):
    routes_dict = {'afi': afi, 'routes': []}
    if isinstance(route_path, dict):
        route_path = [route_path]
    for route in route_path:
        route_dict = {}
        route_dict['dest'] = route['name']
        if route.get('metric'):
            route_dict['metric'] = route['metric']['metric-value']
        route_dict['next_hop'] = []
        route_dict['next_hop'].append({'forward_router_address': route['next-hop']})
        routes_dict['routes'].append(route_dict)
    return routes_dict