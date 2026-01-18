from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def update_device_credentials(self):
    """
        Update Device Credential to the Cisco DNA Center based on the provided playbook details.
        Check the return value of the API with check_return_status().

        Parameters:
            self

        Returns:
            self
        """
    result_global_credential = self.result.get('response')[0].get('globalCredential')
    want_update = self.want.get('want_update')
    if not want_update:
        result_global_credential.update({'No Updation': {'response': 'No Response', 'msg': 'No Updation is available'}})
        self.msg = 'No Updation is available'
        self.status = 'success'
        return self
    i = 0
    flag = True
    values = ['cliCredential', 'snmpV2cRead', 'snmpV2cWrite', 'httpsRead', 'httpsWrite', 'snmpV3']
    final_response = []
    self.log('Desired State for global device credentials updation: {0}'.format(want_update), 'DEBUG')
    while flag:
        flag = False
        credential_params = {}
        for value in values:
            if want_update.get(value) and i < len(want_update.get(value)):
                flag = True
                credential_params.update({value: want_update.get(value)[i]})
        i = i + 1
        if credential_params:
            final_response.append(credential_params)
            response = self.dnac._exec(family='discovery', function='update_global_credentials_v2', params=credential_params)
            self.log("Received API response for 'update_global_credentials_v2': {0}".format(response), 'DEBUG')
            validation_string = 'global credential update performed'
            self.check_task_response_status(response, validation_string).check_return_status()
    self.log('Updating device credential API input parameters: {0}'.format(final_response), 'DEBUG')
    self.log('Global device credential updated successfully', 'INFO')
    result_global_credential.update({'Updation': {'response': final_response, 'msg': 'Global Device Credential Updated Successfully'}})
    self.msg = 'Global Device Credential Updated Successfully'
    self.status = 'success'
    return self