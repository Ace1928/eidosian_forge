from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def _subnet_of(network_a, network_b):
    """Test if a network is a subnet of another network"""
    params = {'network_a': network_a, 'network_b': network_b}
    _validate_args('subnet_of', DOCUMENTATION, params)
    try:
        return _is_subnet_of(ip_network(network_a), ip_network(network_b))
    except Exception:
        return False