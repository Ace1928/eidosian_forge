from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def unassign_partitions(self, required_partitions):
    """
        Unassign partitions from node
        """
    api = 'private/cli/storage/disk/partition/removeowner'
    for required_partition in required_partitions:
        body = {'partition': required_partition['partition']}
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)