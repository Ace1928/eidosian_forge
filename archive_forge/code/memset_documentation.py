from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import json
import ansible.module_utils.six.moves.urllib.error as urllib_error

    Returns the zone's id if it exists and is unique
    