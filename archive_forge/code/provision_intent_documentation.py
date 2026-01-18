from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (

        Delete from provision database
        Args:
            self: An instance of a class used for interacting with Cisco DNA Center
        Returns:
            self: An instance of the class with updated results and status based on
            the deletion operation.
        Description:
            This function is responsible for removing devices from the Cisco DNA Center PnP GUI and
            raise Exception if any error occured.
        