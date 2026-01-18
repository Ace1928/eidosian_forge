from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def wait_for_redfish_reboot_job(redfish_obj, res_id, payload=None, wait_time_sec=300):
    reset, job_resp, msg = (False, {}, '')
    try:
        resp = redfish_obj.invoke_request('POST', SYSTEM_RESET_URI.format(res_id=res_id), data=payload, api_timeout=120)
        time.sleep(10)
        if wait_time_sec and resp.status_code == 204:
            resp = redfish_obj.invoke_request('GET', MANAGER_JOB_URI)
            reboot_job_lst = list(filter(lambda d: d['JobType'] in ['RebootNoForce'], resp.json_data['Members']))
            job_resp = max(reboot_job_lst, key=lambda d: datetime.strptime(d['StartTime'], '%Y-%m-%dT%H:%M:%S'))
            if job_resp:
                reset = True
            else:
                msg = RESET_FAIL
    except Exception:
        reset = False
    return (job_resp, reset, msg)