from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def validate_create_baseline_idempotency(module, rest_obj):
    """
    Idempotency check for compliance baseline create.
    Return error message if baseline name already exists in the system
    """
    name = module.params['names'][0]
    baseline_info = get_baseline_compliance_info(rest_obj, name, attribute='Name')
    if any(baseline_info):
        module.exit_json(msg=BASELINE_CHECK_MODE_CHANGE_MSG.format(name=name), changed=False)
    if not any(baseline_info) and module.check_mode:
        module.exit_json(msg=CHECK_MODE_CHANGES_MSG, changed=True)