from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def validate_vs_registry(registry, attr_dict):
    invalid = {}
    for k, v in attr_dict.items():
        if k in registry:
            val_dict = registry.get(k)
            if val_dict.get('Readonly'):
                invalid[k] = 'Read only Attribute cannot be modified.'
            else:
                type = val_dict.get('Type')
                if type == 'Enumeration':
                    found = False
                    for val in val_dict.get('Value', []):
                        if v == val.get('ValueDisplayName'):
                            found = True
                            break
                    if not found:
                        invalid[k] = 'Invalid value for Enumeration.'
                if type == 'Integer':
                    try:
                        i = int(v)
                    except Exception:
                        invalid[k] = 'Not a valid integer.'
                    else:
                        if not val_dict.get('LowerBound') <= i <= val_dict.get('UpperBound'):
                            invalid[k] = 'Integer out of valid range.'
        else:
            invalid[k] = 'Attribute does not exist.'
    return invalid