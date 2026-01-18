from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def track_log_entry(redfish_obj):
    msg = None
    filter_list = [LC_LOG_FILTER, CPU_RST_FILTER]
    intrvl = 15
    retries = 360 // intrvl
    time.sleep(intrvl)
    try:
        resp = redfish_obj.invoke_request(LOG_SERVICE_URI, 'GET')
        uri = resp.json_data.get('Entries').get('@odata.id')
        fltr_uris = []
        for fltr in filter_list:
            fltr_uris.append('{0}{1}'.format(uri, fltr))
        flen = len(fltr_uris)
        fln = 1
        pvt = retries // 3
        curr_time = resp.json_data.get('DateTime')
        while retries:
            resp = redfish_obj.invoke_request(fltr_uris[retries % fln], 'GET')
            logs_list = resp.json_data.get('Members')
            for log in logs_list:
                if log.get('Created') > curr_time:
                    msg = BIOS_RESET_COMPLETE
                    break
            if msg:
                break
            retries = retries - 1
            time.sleep(intrvl)
            if retries < pvt:
                fln = flen
        else:
            msg = BIOS_RESET_TRIGGERED
    except Exception:
        msg = BIOS_RESET_TRIGGERED
    return msg