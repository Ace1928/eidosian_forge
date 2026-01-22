import threading
from openstack import exceptions
def _reset_network_caches(self):
    with self._networks_lock:
        self._external_ipv4_networks = []
        self._external_ipv4_floating_networks = []
        self._internal_ipv4_networks = []
        self._external_ipv6_networks = []
        self._internal_ipv6_networks = []
        self._nat_destination_network = None
        self._nat_source_network = None
        self._default_network_network = None
        self._network_list_stamp = False