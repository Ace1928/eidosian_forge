from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def update_cvo_gcp(self, working_environment_id, modify):
    base_url = '%s/working-environments/%s/' % (self.rest_api.api_root_path, working_environment_id)
    for item in modify:
        if item == 'svm_password':
            response, error = self.na_helper.update_svm_password(base_url, self.rest_api, self.headers, self.parameters['svm_password'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'svm_name':
            response, error = self.na_helper.update_svm_name(base_url, self.rest_api, self.headers, self.parameters['svm_name'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'gcp_labels':
            tag_list = None
            if 'gcp_labels' in self.parameters:
                tag_list = self.parameters['gcp_labels']
            response, error = self.na_helper.update_cvo_tags(base_url, self.rest_api, self.headers, 'gcp_labels', tag_list)
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'tier_level':
            response, error = self.na_helper.update_tier_level(base_url, self.rest_api, self.headers, self.parameters['tier_level'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'writing_speed_state':
            response, error = self.na_helper.update_writing_speed_state(base_url, self.rest_api, self.headers, self.parameters['writing_speed_state'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'ontap_version':
            response, error = self.na_helper.upgrade_ontap_image(self.rest_api, self.headers, self.parameters['ontap_version'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if item == 'instance_type' or item == 'license_type':
            response, error = self.na_helper.update_instance_license_type(base_url, self.rest_api, self.headers, self.parameters['instance_type'], self.parameters['license_type'])
            if error is not None:
                self.module.fail_json(changed=False, msg=error)