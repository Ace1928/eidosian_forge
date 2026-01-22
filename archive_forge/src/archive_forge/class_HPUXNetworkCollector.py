from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import Network, NetworkCollector
class HPUXNetworkCollector(NetworkCollector):
    _fact_class = HPUXNetwork
    _platform = 'HP-UX'