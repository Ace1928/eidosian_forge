from __future__ import (absolute_import, division, print_function)
import json
import copy
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def _validate_image_format(module):
    unsup_image = False
    for each in module.params['virtual_media']:
        if each['insert'] and each.get('image') is not None and (each.get('image')[-4:].lower() not in ['.iso', '.img']):
            unsup_image = True
    if unsup_image:
        module.fail_json(msg=UNSUPPORTED_IMAGE)