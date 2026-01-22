from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def add_field_to_devices(self, device_ids, udf):
    """
        Add a Global user-defined field with specified details to a list of devices in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_ids (list): A list of device IDs to which the user-defined field will be added.
            udf (dict): A dictionary having the user defined field details including name and value.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            The function retrieves the details of the user-defined field from the configuration object,
            including the field name and default value then iterates over list of device IDs, creating a payload for
            each device and sending the request to Cisco Catalyst Center to add the user-defined field.
        """
    field_name = udf.get('name')
    field_value = udf.get('value', '1')
    for device_id in device_ids:
        payload = {}
        payload['name'] = field_name
        payload['value'] = field_value
        udf_param_dict = {'payload': [payload], 'device_id': device_id}
        try:
            response = self.dnac._exec(family='devices', function='add_user_defined_field_to_device', params=udf_param_dict)
            self.log("Received API response from 'add_user_defined_field_to_device': {0}".format(str(response)), 'DEBUG')
            response = response.get('response')
            self.status = 'success'
            self.result['changed'] = True
        except Exception as e:
            self.status = 'failed'
            error_message = 'Error while adding Global UDF to device in Cisco Catalyst Center: {0}'.format(str(e))
            self.log(error_message, 'ERROR')
            self.result['changed'] = False
    return self