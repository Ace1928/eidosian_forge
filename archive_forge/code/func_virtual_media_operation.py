from __future__ import (absolute_import, division, print_function)
import json
import copy
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def virtual_media_operation(idrac, module, payload, vr_id):
    err_payload = []
    force = module.params['force']
    for i in payload:
        try:
            if force and i['vr_mem']['Inserted'] and i['payload']['Inserted']:
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.EjectMedia']['target'], 'POST', data='{}', dump=False)
                time.sleep(5)
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.InsertMedia']['target'], 'POST', data=i['payload'])
            elif not force and i['vr_mem']['Inserted'] and i['payload']['Inserted']:
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.EjectMedia']['target'], 'POST', data='{}', dump=False)
                time.sleep(5)
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.InsertMedia']['target'], 'POST', data=i['payload'])
            elif not i['vr_mem']['Inserted'] and i['payload']['Inserted']:
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.InsertMedia']['target'], 'POST', data=i['payload'])
            elif i['vr_mem']['Inserted'] and (not i['payload']['Inserted']):
                idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.EjectMedia']['target'], 'POST', data='{}', dump=False)
            time.sleep(5)
        except Exception as err:
            error = json.load(err).get('error')
            if vr_id == 'manager':
                msg_id = error['@Message.ExtendedInfo'][0]['MessageId']
                if 'VRM0021' in msg_id or 'VRM0012' in msg_id:
                    uri = i['vr_mem']['Actions']['#VirtualMedia.EjectMedia']['target']
                    if 'RemovableDisk' in uri:
                        uri = uri.replace('RemovableDisk', 'CD')
                    elif 'CD' in uri:
                        uri = uri.replace('CD', 'RemovableDisk')
                    idrac.invoke_request(uri, 'POST', data='{}', dump=False)
                    time.sleep(5)
                    idrac.invoke_request(i['vr_mem']['Actions']['#VirtualMedia.InsertMedia']['target'], 'POST', data=i['payload'])
                else:
                    err_payload.append(error)
            else:
                err_payload.append(error)
    return err_payload