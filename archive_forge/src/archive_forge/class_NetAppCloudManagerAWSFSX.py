from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
class NetAppCloudManagerAWSFSX:
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.cloudmanager_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=True, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), region=dict(required=False, type='str'), aws_credentials_name=dict(required=False, type='str'), workspace_id=dict(required=False, type='str'), tenant_id=dict(required=True, type='str'), working_environment_id=dict(required=False, type='str'), storage_capacity_size=dict(required=False, type='int'), storage_capacity_size_unit=dict(required=False, type='str', choices=['GiB', 'TiB']), fsx_admin_password=dict(required=False, type='str', no_log=True), throughput_capacity=dict(required=False, type='int', choices=[512, 1024, 2048]), security_group_ids=dict(required=False, type='list', elements='str'), kms_key_id=dict(required=False, type='str', no_log=True), tags=dict(required=False, type='list', elements='dict', options=dict(tag_key=dict(type='str', no_log=False), tag_value=dict(type='str'))), primary_subnet_id=dict(required=False, type='str'), secondary_subnet_id=dict(required=False, type='str'), route_table_ids=dict(required=False, type='list', elements='str'), minimum_ssd_iops=dict(required=False, type='int'), endpoint_ip_address_range=dict(required=False, type='str'), import_file_system=dict(required=False, type='bool', default=False), file_system_id=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[['state', 'present', ['region', 'aws_credentials_name', 'workspace_id', 'fsx_admin_password', 'throughput_capacity', 'primary_subnet_id', 'secondary_subnet_id', 'storage_capacity_size', 'storage_capacity_size_unit']], ['import_file_system', True, ['file_system_id']]], required_one_of=[['refresh_token', 'sa_client_id']], required_together=[['sa_client_id', 'sa_secret_key'], ['storage_capacity_size', 'storage_capacity_size_unit']], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = CloudManagerRestAPI(self.module)
        self.rest_api.url += self.rest_api.environment_data['CLOUD_MANAGER_HOST']
        self.headers = None
        if self.rest_api.simulator:
            self.headers = {'x-simulator': 'true'}
        if self.parameters['state'] == 'present':
            self.aws_credentials_id, error = self.get_aws_credentials_id()
            if error is not None:
                self.module.fail_json(msg=str(error))

    def get_aws_credentials_id(self):
        """
        Get aws_credentials_id
        :return: AWS Credentials ID
        """
        api = '/fsx-ontap/aws-credentials/'
        api += self.parameters['tenant_id']
        response, error, dummy = self.rest_api.get(api, None, header=self.headers)
        if error:
            return (response, 'Error: getting aws_credentials_id %s' % error)
        for each in response:
            if each['name'] == self.parameters['aws_credentials_name']:
                return (each['id'], None)
        return (None, 'Error: aws_credentials_name not found')

    def discover_aws_fsx(self):
        """
        discover aws_fsx
        """
        api = '/fsx-ontap/working-environments/%s/discover?credentials-id=%s&workspace-id=%s&region=%s' % (self.parameters['tenant_id'], self.aws_credentials_id, self.parameters['workspace_id'], self.parameters['region'])
        response, error, dummy = self.rest_api.get(api, None, header=self.headers)
        if error:
            return 'Error: discovering aws_fsx %s' % error
        id_found = False
        for each in response:
            if each['id'] == self.parameters['file_system_id']:
                id_found = True
                break
        if not id_found:
            return 'Error: file_system_id provided could not be found'

    def recover_aws_fsx(self):
        """
        recover aws_fsx
        """
        json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'workspaceId': self.parameters['workspace_id'], 'credentialsId': self.aws_credentials_id, 'fileSystemId': self.parameters['file_system_id']}
        api_url = '/fsx-ontap/working-environments/%s/recover' % self.parameters['tenant_id']
        response, error, dummy = self.rest_api.post(api_url, json, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on recovering AWS FSx: %s, %s' % (error, response))

    def create_aws_fsx(self):
        """ Create AWS FSx """
        json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'workspaceId': self.parameters['workspace_id'], 'credentialsId': self.aws_credentials_id, 'throughputCapacity': self.parameters['throughput_capacity'], 'storageCapacity': {'size': self.parameters['storage_capacity_size'], 'unit': self.parameters['storage_capacity_size_unit']}, 'fsxAdminPassword': self.parameters['fsx_admin_password'], 'primarySubnetId': self.parameters['primary_subnet_id'], 'secondarySubnetId': self.parameters['secondary_subnet_id']}
        if self.parameters.get('tags') is not None:
            tags = []
            for each_tag in self.parameters['tags']:
                tag = {'key': each_tag['tag_key'], 'value': each_tag['tag_value']}
                tags.append(tag)
            json.update({'tags': tags})
        if self.parameters.get('security_group_ids'):
            json.update({'securityGroupIds': self.parameters['security_group_ids']})
        if self.parameters.get('route_table_ids'):
            json.update({'routeTableIds': self.parameters['route_table_ids']})
        if self.parameters.get('kms_key_id'):
            json.update({'kmsKeyId': self.parameters['kms_key_id']})
        if self.parameters.get('minimum_ssd_iops'):
            json.update({'minimumSsdIops': self.parameters['minimum_ssd_iops']})
        if self.parameters.get('endpoint_ip_address_range'):
            json.update({'endpointIpAddressRange': self.parameters['endpoint_ip_address_range']})
        api_url = '/fsx-ontap/working-environments/%s' % self.parameters['tenant_id']
        response, error, dummy = self.rest_api.post(api_url, json, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on creating AWS FSx: %s, %s' % (str(error), str(response)))
        working_environment_id = response['id']
        creation_wait_time = 30
        creation_retry_count = 30
        wait_on_completion_api_url = '/fsx-ontap/working-environments/%s/%s?provider-details=true' % (self.parameters['tenant_id'], working_environment_id)
        err = self.wait_on_completion_for_fsx(wait_on_completion_api_url, 'AWS_FSX', 'create', creation_retry_count, creation_wait_time)
        if err is not None:
            self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating AWS FSX: %s' % str(err))
        return working_environment_id

    def wait_on_completion_for_fsx(self, api_url, action_name, task, retries, wait_interval):
        while True:
            fsx_status, error = self.check_task_status_for_fsx(api_url)
            if error is not None:
                return error
            if fsx_status['status']['status'] == 'ON' and fsx_status['status']['lifecycle'] == 'AVAILABLE':
                return None
            elif fsx_status['status']['status'] == 'FAILED':
                return 'Failed to %s %s' % (task, action_name)
            if retries == 0:
                return 'Taking too long for %s to %s or not properly setup' % (action_name, task)
            time.sleep(wait_interval)
            retries = retries - 1

    def check_task_status_for_fsx(self, api_url):
        network_retries = 3
        exponential_retry_time = 1
        while True:
            result, error, dummy = self.rest_api.get(api_url, None, header=self.headers)
            if error is not None:
                if network_retries > 0:
                    time.sleep(exponential_retry_time)
                    exponential_retry_time *= 2
                    network_retries = network_retries - 1
                else:
                    return (0, error)
            else:
                response = result
                break
        return (response['providerDetails'], None)

    def delete_aws_fsx(self, id, tenant_id):
        """
        Delete AWS FSx
        """
        api_url = '/fsx-ontap/working-environments/%s/%s' % (tenant_id, id)
        response, error, dummy = self.rest_api.delete(api_url, None, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on deleting AWS FSx: %s, %s' % (str(error), str(response)))

    def apply(self):
        """
        Apply action to the AWS FSx working Environment
        :return: None
        """
        working_environment_id = None
        current, error = self.na_helper.get_aws_fsx_details(self.rest_api, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on fetching AWS FSx: %s' % str(error))
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.parameters['import_file_system'] and cd_action == 'create':
            error = self.discover_aws_fsx()
            if error is not None:
                self.module.fail_json(msg='Error: unexpected response on discovering AWS FSx: %s' % str(error))
            cd_action = 'import'
            self.na_helper.changed = True
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'import':
                self.recover_aws_fsx()
                working_environment_id = self.parameters['file_system_id']
            elif cd_action == 'create':
                working_environment_id = self.create_aws_fsx()
            elif cd_action == 'delete':
                self.delete_aws_fsx(current['id'], self.parameters['tenant_id'])
        self.module.exit_json(changed=self.na_helper.changed, working_environment_id=working_environment_id)