from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def unassign_disks(self, disks):
    """
        Unassign disks.
        Disk autoassign must be turned off when removing ownership of a disk
        """
    api = 'private/cli/storage/disk/removeowner'
    for disk in disks:
        body = {'disk': disk['name']}
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)