from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_IPV4_NETWORK, NIOS_IPV6_NETWORK
from ..module_utils.api import NIOS_IPV4_NETWORK_CONTAINER, NIOS_IPV6_NETWORK_CONTAINER
from ..module_utils.api import normalize_ib_spec
from ..module_utils.network import validate_ip_address, validate_ip_v6_address
 Main entry point for module execution
    