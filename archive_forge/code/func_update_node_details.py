from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
import copy
def update_node_details(self, uuid, modify):
    api = 'cluster/nodes/%s' % uuid
    data = {}
    if 'from_name' in self.parameters:
        data['name'] = self.parameters['name']
    if 'location' in self.parameters:
        data['location'] = self.parameters['location']
    if not data:
        self.module.fail_json(msg='Nothing to update in the modified attributes: %s' % modify)
    response, error = self.rest_api.patch(api, body=data)
    response, error = rrh.check_for_error_and_job_results(api, response, error, self.rest_api)
    if error:
        self.module.fail_json(msg='Error while modifying node details: %s' % error)