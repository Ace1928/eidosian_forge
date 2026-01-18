from __future__ import (absolute_import, division, print_function)
import socket
import json
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError

    Export Lifecycle Controller Log to the given file share

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    