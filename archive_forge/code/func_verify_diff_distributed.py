from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def verify_diff_distributed(self):
    """
        Verify the distribution status of a software image in Cisco Catalyst Center.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            import_type (str): The type of import, either 'url' or 'local'.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This method verifies the distribution status of a software image in Cisco Catalyst Center.
            It retrieves the image ID and name from the input and if distribution device ID is provided, it checks the distribution status for that
            list of specific device and logs the info message based on distribution status.
        """
    image_id = self.have.get('distribution_image_id')
    image_name = self.get_image_name_from_id(image_id)
    if self.have.get('distribution_device_id'):
        if self.single_device_distribution:
            self.msg = "The requested image '{0}', associated with the device ID '{1}', has been successfully distributed in the Cisco Catalyst Center\n                     and its status has been verified.".format(image_name, self.have.get('distribution_device_id'))
            self.log(self.msg, 'INFO')
        else:
            self.log("Mismatch between the playbook input for distributing the image to the device with ID '{0}' and the actual state in the\n                         Cisco Catalyst Center suggests that the distribution task might not have been executed\n                         successfully.".format(self.have.get('distribution_device_id')), 'INFO')
    elif self.complete_successful_distribution:
        self.msg = "The requested image '{0}', with ID '{1}', has been successfully distributed to all devices within the specified\n                     site in the Cisco Catalyst Center.".format(image_name, image_id)
        self.log(self.msg, 'INFO')
    elif self.partial_successful_distribution:
        self.msg = 'T"The requested image \'{0}\', with ID \'{1}\', has been partially distributed across some devices in the Cisco Catalyst\n                     Center.'.format(image_name, image_id)
        self.log(self.msg, 'INFO')
    else:
        self.msg = "The requested image '{0}', with ID '{1}', failed to be distributed across devices in the Cisco Catalyst\n                     Center.".format(image_name, image_id)
        self.log(self.msg, 'INFO')
    return self