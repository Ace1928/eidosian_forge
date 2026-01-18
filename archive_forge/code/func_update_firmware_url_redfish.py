from __future__ import (absolute_import, division, print_function)
import os
import json
import time
from ssl import SSLError
from xml.etree import ElementTree as ET
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def update_firmware_url_redfish(module, idrac, share_path, apply_update, reboot, job_wait, payload, repo_urls):
    """Update firmware through HTTP/HTTPS/FTP and return the job details."""
    repo_url = urlparse(share_path)
    job_details, status = ({}, {})
    ipaddr = repo_url.netloc
    share_type = repo_url.scheme
    sharename = repo_url.path.strip('/')
    if repo_url.path:
        payload['ShareName'] = sharename
    payload['IPAddress'] = ipaddr
    payload['ShareType'] = SHARE_TYPE[share_type]
    install_url = PATH
    get_repo_url = GET_REPO_BASED_UPDATE_LIST_PATH
    actions = repo_urls.get('Actions')
    if actions:
        install_url = actions.get('#DellSoftwareInstallationService.InstallFromRepository', {}).get('target', PATH)
        get_repo_url = actions.get('#DellSoftwareInstallationService.GetRepoBasedUpdateList', {}).get('target', GET_REPO_BASED_UPDATE_LIST_PATH)
    try:
        log_resp = idrac.invoke_request(LOG_SERVICE_URI, 'GET')
        log_uri = log_resp.json_data.get('Entries').get('@odata.id')
        curr_time = log_resp.json_data.get('DateTime')
    except Exception:
        log_uri = iDRAC9_LC_LOG
        curr_time = None
    resp = idrac.invoke_request(install_url, method='POST', data=payload)
    error_log_found, msg = get_error_syslog(idrac, curr_time, log_uri)
    job_id = get_jobid(module, resp)
    if error_log_found:
        module.exit_json(msg=msg, failed=True, job_id=job_id)
    resp, msg = wait_for_job_completion(module, JOB_URI.format(job_id=job_id), job_wait, reboot, apply_update)
    if not msg:
        status = resp.json_data
    else:
        status['update_msg'] = msg
    try:
        resp_repo_based_update_list = idrac.invoke_request(get_repo_url, method='POST', data='{}', dump=False)
        job_details = resp_repo_based_update_list.json_data
    except HTTPError as err:
        handle_HTTP_error(module, err)
        raise err
    return (status, job_details)