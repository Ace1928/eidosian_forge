from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.network.a10.a10 import (axapi_call, a10_argument_spec, axapi_authenticate, axapi_failure,
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec

                Checks to determine if the port definitions of the src_ports
                array are in or different from those in dst_ports. If there is
                a difference, this function returns true, otherwise false.
                