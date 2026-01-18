from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def verify_diff_imported(self, import_type):
    """
        Verify the successful import of a software image into Cisco Catalyst Center.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            import_type (str): The type of import, either 'remote' or 'local'.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This method verifies the successful import of a software image into Cisco Catalyst Center.
            It checks whether the image exists in Catalyst Center based on the provided import type.
            If the image exists, the status is set to 'success', and a success message is logged.
            If the image does not exist, a warning message is logged indicating a potential import failure.
        """
    if import_type == 'remote':
        image_name = self.want.get('url_import_details').get('payload')[0].get('source_url')
    else:
        image_name = self.want.get('local_import_details').get('file_path')
    name = image_name.split('/')[-1]
    image_exist = self.is_image_exist(name)
    if image_exist:
        self.status = 'success'
        self.msg = "The requested Image '{0}' imported in the Cisco Catalyst Center and Image presence has been verified.".format(name)
        self.log(self.msg, 'INFO')
    else:
        self.log("The playbook input for SWIM Image '{0}' does not align with the Cisco Catalyst Center, indicating that image\n                        may not have imported successfully.".format(name), 'INFO')
    return self