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
def update_firmware_omsdk(idrac, module):
    """Update firmware from a network share and return the job details."""
    msg = {}
    msg['changed'], msg['failed'], msg['update_status'] = (False, False, {})
    msg['update_msg'] = 'Successfully triggered the job to update the firmware.'
    try:
        share_name = module.params['share_name']
        catalog_file_name = module.params['catalog_file_name']
        share_user = module.params['share_user']
        share_pwd = module.params['share_password']
        reboot = module.params['reboot']
        job_wait = module.params['job_wait']
        ignore_cert_warning = module.params['ignore_cert_warning']
        apply_update = module.params['apply_update']
        payload = {'RebootNeeded': reboot, 'CatalogFile': catalog_file_name, 'ApplyUpdate': str(apply_update), 'IgnoreCertWarning': CERT_WARN[ignore_cert_warning]}
        if share_user is not None:
            payload['UserName'] = share_user
        if share_pwd is not None:
            payload['Password'] = share_pwd
        if share_name.lower().startswith(('http://', 'https://', 'ftp://')):
            msg['update_status'], job_details = update_firmware_url_omsdk(module, idrac, share_name, catalog_file_name, apply_update, reboot, ignore_cert_warning, job_wait, payload)
            if job_details:
                msg['update_status']['job_details'] = job_details
        else:
            upd_share = FileOnShare(remote='{0}{1}{2}'.format(share_name, os.sep, catalog_file_name), mount_point=module.params['share_mnt'], isFolder=False, creds=UserCredentials(share_user, share_pwd))
            msg['update_status'] = idrac.update_mgr.update_from_repo(upd_share, apply_update=apply_update, reboot_needed=reboot, job_wait=job_wait)
            get_check_mode_status(msg['update_status'], module)
        json_data, repo_status, failed = (msg['update_status']['job_details'], False, False)
        if 'PackageList' not in json_data:
            job_data = json_data.get('Data')
            pkglst = job_data['body'] if 'body' in job_data else job_data.get('GetRepoBasedUpdateList_OUTPUT')
            if 'PackageList' in pkglst:
                pkglst['PackageList'], repo_status, failed = _convert_xmltojson(module, pkglst, idrac)
        else:
            json_data['PackageList'], repo_status, failed = _convert_xmltojson(module, json_data, None)
        if not apply_update and (not failed):
            msg['update_msg'] = 'Successfully fetched the applicable firmware update package list.'
        elif apply_update and (not reboot) and (not job_wait) and (not failed):
            msg['update_msg'] = 'Successfully triggered the job to stage the firmware.'
        elif apply_update and job_wait and (not reboot) and (not failed):
            msg['update_msg'] = 'Successfully staged the applicable firmware update packages.'
            msg['changed'] = True
        elif apply_update and job_wait and (not reboot) and failed:
            msg['update_msg'] = 'Successfully staged the applicable firmware update packages with error(s).'
            msg['failed'] = True
    except RuntimeError as e:
        module.fail_json(msg=str(e))
    if module.check_mode and (not (json_data.get('PackageList') or json_data.get('Data'))) and (msg['update_status']['JobStatus'] == 'Completed'):
        module.exit_json(msg='No changes found to commit!')
    elif module.check_mode and (json_data.get('PackageList') or json_data.get('Data')) and (msg['update_status']['JobStatus'] == 'Completed'):
        module.exit_json(msg='Changes found to commit!', changed=True, update_status=msg['update_status'])
    elif module.check_mode and (not msg['update_status']['JobStatus'] == 'Completed'):
        msg['update_status'].pop('job_details')
        module.fail_json(msg='Unable to complete the firmware repository download.', update_status=msg['update_status'])
    elif not module.check_mode and 'Status' in msg['update_status']:
        if msg['update_status']['Status'] in ['Success', 'InProgress']:
            if module.params['job_wait'] and module.params['apply_update'] and module.params['reboot'] and ('job_details' in msg['update_status'] and repo_status) and (not failed):
                msg['changed'] = True
                msg['update_msg'] = 'Successfully updated the firmware.'
            elif module.params['job_wait'] and module.params['apply_update'] and module.params['reboot'] and ('job_details' in msg['update_status'] and repo_status) and failed:
                msg['failed'], msg['changed'] = (True, False)
                msg['update_msg'] = 'Firmware update failed.'
        else:
            failed_msg = 'Firmware update failed.'
            if not apply_update:
                failed_msg = 'Unable to complete the repository update.'
            module.fail_json(msg=failed_msg, update_status=msg['update_status'])
    return msg