from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def validate_modify(module, current_payload):
    """Fabric modification does not support fabric design type modification"""
    if module.params.get('fabric_design') and current_payload['FabricDesign']['Name'] and (module.params.get('fabric_design') != current_payload['FabricDesign']['Name']):
        module.fail_json(msg='The modify operation does not support fabric_design update.')