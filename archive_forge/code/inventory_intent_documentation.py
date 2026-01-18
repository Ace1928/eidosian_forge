from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (

            Get the interface ID for a device in Cisco Catalyst Center based on its IP address.
            Parameters:
                self (object): An instance of a class used for interacting with Cisco Catalyst Center.
                device_ip (str): The IP address of the device.
            Returns:
                str: The interface ID for the specified device.
            Description:
                The function sends a request to Cisco Catalyst Center to retrieve the interface information
                for the device with the provided IP address and extracts the interface ID from the
                response, and returns the interface ID.
            