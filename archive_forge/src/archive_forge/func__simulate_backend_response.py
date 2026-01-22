from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ..module_utils import arguments, errors, utils
def _simulate_backend_response(payload):
    masked_keys = ('password', 'password_hash')
    return dict(((k, v) for k, v in payload.items() if k not in masked_keys))