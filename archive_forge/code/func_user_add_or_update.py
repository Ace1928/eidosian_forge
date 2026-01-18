from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def user_add_or_update(module, HTTPMethod, existing_info):
    username = module.params['username']
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    result = {}
    payload = {}
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    if HTTPMethod == 'POST':
        end_point = 'PasswordVault/api/Users'
        payload['UserName'] = username
        if 'initial_password' in list(module.params.keys()) and module.params['initial_password'] is not None:
            payload['InitialPassword'] = module.params['initial_password']
    elif HTTPMethod == 'PUT':
        end_point = '/PasswordVault/WebServices/PIMServices.svc/Users/{pusername}'.format(pusername=username)
    if 'new_password' in module.params and module.params['new_password'] is not None:
        payload['NewPassword'] = module.params['new_password']
    if 'email' in module.params and module.params['email'] is not None:
        payload['Email'] = module.params['email']
    if 'first_name' in module.params and module.params['first_name'] is not None:
        payload['FirstName'] = module.params['first_name']
    if 'last_name' in module.params and module.params['last_name'] is not None:
        payload['LastName'] = module.params['last_name']
    if 'change_password_on_the_next_logon' in module.params and module.params['change_password_on_the_next_logon'] is not None:
        payload['ChangePasswordOnTheNextLogon'] = module.params['change_password_on_the_next_logon']
    if 'expiry_date' in module.params and module.params['expiry_date'] is not None:
        payload['ExpiryDate'] = module.params['expiry_date']
    if 'user_type_name' in module.params and module.params['user_type_name'] is not None:
        payload['UserTypeName'] = module.params['user_type_name']
        payload['userType'] = module.params['user_type_name']
    if 'disabled' in module.params and module.params['disabled'] is not None:
        payload['Disabled'] = module.params['disabled']
    if 'location' in module.params and module.params['location'] is not None:
        payload['Location'] = module.params['location']
    if module.params.get('authorization', None) is not None:
        payload['vaultAuthorization'] = module.params['authorization']
    logging.debug('HTTPMethod = ' + HTTPMethod + ' module.params = ' + json.dumps(module.params))
    logging.debug('Existing Info: %s', json.dumps(existing_info))
    logging.debug('payload => %s', json.dumps(payload))
    if HTTPMethod == 'PUT' and ('new_password' not in module.params or module.params['new_password'] is None):
        logging.info('Verifying if needs to be updated')
        proceed = False
        updateable_fields = ['Email', 'FirstName', 'LastName', 'ChangePasswordOnTheNextLogon', 'ExpiryDate', 'UserTypeName', 'Disabled', 'Location', 'UserTypeName', 'vaultAuthorization']
        for field_name in updateable_fields:
            logging.debug('#### field_name : %s', field_name)
            if field_name in payload and field_name in existing_info and (payload[field_name] != existing_info[field_name]):
                logging.debug('Changing value for %s', field_name)
                proceed = True
    else:
        proceed = True
    if proceed:
        logging.info('Proceeding to either update or create')
        url = construct_url(api_base_url, end_point)
        try:
            response = open_url(url, method=HTTPMethod, headers=headers, data=json.dumps(payload), validate_certs=validate_certs, timeout=module.params['timeout'])
            result = {'result': json.loads(response.read())}
            return (True, result, response.getcode())
        except (HTTPError, httplib.HTTPException) as http_exception:
            module.fail_json(msg='Error while performing user_add_or_update.Please validate parameters provided.\n*** end_point=%s\n ==> %s' % (url, to_text(http_exception)), payload=payload, headers=headers, status_code=http_exception.code)
        except Exception as unknown_exception:
            module.fail_json(msg='Unknown error while performing user_add_or_update.\n*** end_point=%s\n%s' % (url, to_text(unknown_exception)), payload=payload, headers=headers, status_code=-1)
    else:
        return (False, existing_info, 200)