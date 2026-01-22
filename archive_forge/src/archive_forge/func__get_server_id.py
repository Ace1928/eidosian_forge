from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _get_server_id(servers, identity):
    """
    Fetch and return server UUID by server name if found.
    """
    for server in servers['items']:
        if identity in (server['properties']['name'], server['id']):
            return server['id']
    return None