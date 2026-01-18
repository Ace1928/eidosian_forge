from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def validate_job_time(command, module):
    """
    The command create, remediate and modify time validation
    """
    job_wait = module.params['job_wait']
    if command != 'delete' and job_wait:
        job_wait_timeout = module.params['job_wait_timeout']
        if job_wait_timeout <= 0:
            module.fail_json(msg=INVALID_TIME.format(job_wait_timeout))