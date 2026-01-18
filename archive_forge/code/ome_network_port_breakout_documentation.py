from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError

    Configuration the breakout feature for given option.
    :param module: ansible module arguments.
    :param rest_obj: rest object for making requests.
    :param breakout_config: Existing breakout configuration.
    :param breakout_capability: Available breakout configuration.
    :param interface_id: port number with service tag
    :param device_id: device id
    :return: rest object
    