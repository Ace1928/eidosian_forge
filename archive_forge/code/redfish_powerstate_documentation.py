from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError

    Apply reset type to system
    Keyword arguments:
    redfish_session_obj  -- session handle
    module -- Ansible module obj
    