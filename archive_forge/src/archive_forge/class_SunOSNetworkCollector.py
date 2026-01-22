from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class SunOSNetworkCollector(NetworkCollector):
    _fact_class = SunOSNetwork
    _platform = 'SunOS'