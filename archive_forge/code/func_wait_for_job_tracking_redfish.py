from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def wait_for_job_tracking_redfish(module, idrac, scp_response):
    job_id = scp_response.headers['Location'].split('/')[-1]
    if module.params['job_wait']:
        job_failed, _msg, job_dict, _wait_time = idrac_redfish_job_tracking(idrac, iDRAC_JOB_URI.format(job_id=job_id))
        if job_failed or job_dict.get('MessageId', '') in ERROR_CODES:
            module.exit_json(failed=True, status_msg=job_dict, job_id=job_id, msg=FAIL_MSG.format(module.params['command']))
        scp_response = job_dict
    return scp_response