from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError

    Get device ids from device service tag
    Returns :dict : device_id to service_tag map
    :arg service_tags: service tag
    :arg rest_obj: RestOME class object in case of request with session.
    :returns: dict eg: {1345:"MXL1245"}
    