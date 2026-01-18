from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def validate_group(group_resp, module, identifier, identifier_val):
    if not group_resp:
        module.fail_json(msg="Unable to complete the operation because the entered target group {identifier} '{val}' is invalid.".format(identifier=identifier, val=identifier_val))
    system_groups = group_resp['TypeId']
    membership_id = group_resp['MembershipTypeId']
    if system_groups != 3000 or (system_groups == 3000 and membership_id == 24):
        msg = ADD_STATIC_GROUP_MESSAGE if module.params.get('state', 'present') == 'present' else REMOVE_STATIC_GROUP_MESSAGE
        module.fail_json(msg=msg)