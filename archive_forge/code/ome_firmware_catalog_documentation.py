from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError

    The state present doest not supports more than one name/id
    