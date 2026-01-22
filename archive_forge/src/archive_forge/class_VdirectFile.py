from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
import os
import os.path
class VdirectFile(object):

    def __init__(self, params):
        self.client = rest_client.RestClient(params['vdirect_ip'], params['vdirect_user'], params['vdirect_password'], wait=params['vdirect_wait'], secondary_vdirect_ip=params['vdirect_secondary_ip'], https_port=params['vdirect_https_port'], http_port=params['vdirect_http_port'], timeout=params['vdirect_timeout'], https=params['vdirect_use_ssl'], verify=params['validate_certs'])

    def upload(self, fqn):
        if fqn.endswith(TEMPLATE_EXTENSION):
            template_name = os.path.basename(fqn)
            template = rest_client.Template(self.client)
            runnable_file = open(fqn, 'r')
            file_content = runnable_file.read()
            result_to_return = CONFIGURATION_TEMPLATE_CREATED_SUCCESS
            result = template.create_from_source(file_content, template_name, fail_if_invalid=True)
            if result[rest_client.RESP_STATUS] == 409:
                result_to_return = CONFIGURATION_TEMPLATE_UPDATED_SUCCESS
                result = template.upload_source(file_content, template_name, fail_if_invalid=True)
            if result[rest_client.RESP_STATUS] == 400:
                raise InvalidSourceException(str(result[rest_client.RESP_STR]))
        elif fqn.endswith(WORKFLOW_EXTENSION):
            workflow = rest_client.WorkflowTemplate(self.client)
            runnable_file = open(fqn, 'rb')
            file_content = runnable_file.read()
            result_to_return = WORKFLOW_TEMPLATE_CREATED_SUCCESS
            result = workflow.create_template_from_archive(file_content, fail_if_invalid=True)
            if result[rest_client.RESP_STATUS] == 409:
                result_to_return = WORKFLOW_TEMPLATE_UPDATED_SUCCESS
                result = workflow.update_archive(file_content, os.path.splitext(os.path.basename(fqn))[0])
            if result[rest_client.RESP_STATUS] == 400:
                raise InvalidSourceException(str(result[rest_client.RESP_STR]))
        else:
            result_to_return = WRONG_EXTENSION_ERROR
        return result_to_return