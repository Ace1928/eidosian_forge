from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
class NetAppCloudManagerCVOAWS:
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.cloudmanager_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=True, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), instance_type=dict(required=False, type='str', default='m5.2xlarge'), license_type=dict(required=False, type='str', choices=AWS_License_Types, default='capacity-paygo'), workspace_id=dict(required=False, type='str'), subnet_id=dict(required=False, type='str'), vpc_id=dict(required=False, type='str'), region=dict(required=True, type='str'), data_encryption_type=dict(required=False, type='str', choices=['AWS', 'NONE'], default='AWS'), ebs_volume_size=dict(required=False, type='int', default='1'), ebs_volume_size_unit=dict(required=False, type='str', choices=['GB', 'TB'], default='TB'), ebs_volume_type=dict(required=False, type='str', choices=['gp3', 'gp2', 'io1', 'sc1', 'st1'], default='gp2'), svm_password=dict(required=True, type='str', no_log=True), svm_name=dict(required=False, type='str'), ontap_version=dict(required=False, type='str', default='latest'), use_latest_version=dict(required=False, type='bool', default=True), platform_serial_number=dict(required=False, type='str'), capacity_package_name=dict(required=False, type='str', choices=['Professional', 'Essential', 'Freemium'], default='Essential'), provided_license=dict(required=False, type='str'), tier_level=dict(required=False, type='str', choices=['normal', 'ia', 'ia-single', 'intelligent'], default='normal'), cluster_key_pair_name=dict(required=False, type='str'), nss_account=dict(required=False, type='str'), writing_speed_state=dict(required=False, type='str'), iops=dict(required=False, type='int'), throughput=dict(required=False, type='int'), capacity_tier=dict(required=False, type='str', choices=['S3', 'NONE'], default='S3'), instance_tenancy=dict(required=False, type='str', choices=['default', 'dedicated'], default='default'), instance_profile_name=dict(required=False, type='str'), security_group_id=dict(required=False, type='str'), cloud_provider_account=dict(required=False, type='str'), backup_volumes_to_cbs=dict(required=False, type='bool', default=False), enable_compliance=dict(required=False, type='bool', default=False), enable_monitoring=dict(required=False, type='bool', default=False), optimized_network_utilization=dict(required=False, type='bool', default=True), kms_key_id=dict(required=False, type='str', no_log=True), kms_key_arn=dict(required=False, type='str', no_log=True), client_id=dict(required=True, type='str'), aws_tag=dict(required=False, type='list', elements='dict', options=dict(tag_key=dict(type='str', no_log=False), tag_value=dict(type='str'))), is_ha=dict(required=False, type='bool', default=False), platform_serial_number_node1=dict(required=False, type='str'), platform_serial_number_node2=dict(required=False, type='str'), failover_mode=dict(required=False, type='str', choices=['PrivateIP', 'FloatingIP']), mediator_assign_public_ip=dict(required=False, type='bool', default=True), node1_subnet_id=dict(required=False, type='str'), node2_subnet_id=dict(required=False, type='str'), mediator_subnet_id=dict(required=False, type='str'), mediator_key_pair_name=dict(required=False, type='str'), cluster_floating_ip=dict(required=False, type='str'), data_floating_ip=dict(required=False, type='str'), data_floating_ip2=dict(required=False, type='str'), svm_floating_ip=dict(required=False, type='str'), route_table_ids=dict(required=False, type='list', elements='str'), upgrade_ontap_version=dict(required=False, type='bool', default=False), update_svm_password=dict(required=False, type='bool', default=False)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[['ebs_volume_type', 'gp3', ['iops', 'throughput']], ['ebs_volume_type', 'io1', ['iops']], ['license_type', 'cot-premium-byol', ['platform_serial_number']], ['license_type', 'ha-cot-premium-byol', ['platform_serial_number_node1', 'platform_serial_number_node2']], ['license_type', 'capacity-paygo', ['capacity_package_name']], ['license_type', 'ha-capacity-paygo', ['capacity_package_name']]], required_one_of=[['refresh_token', 'sa_client_id']], mutually_exclusive=[['kms_key_id', 'kms_key_arn']], required_together=[['sa_client_id', 'sa_secret_key']], supports_check_mode=True)
        if HAS_AWS_LIB is False:
            self.module.fail_json(msg='the python AWS library boto3 and botocore is required. Command is pip install boto3.Import error: %s' % str(IMPORT_EXCEPTION))
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.changeable_params = ['aws_tag', 'svm_password', 'svm_name', 'tier_level', 'ontap_version', 'instance_type', 'license_type', 'writing_speed_state']
        self.rest_api = CloudManagerRestAPI(self.module)
        self.rest_api.url += self.rest_api.environment_data['CLOUD_MANAGER_HOST']
        self.rest_api.api_root_path = '/occm/api/%s' % ('aws/ha' if self.parameters['is_ha'] else 'vsa')
        self.headers = {'X-Agent-Id': self.rest_api.format_client_id(self.parameters['client_id'])}

    def get_vpc(self):
        """
        Get vpc
        :return: vpc ID
        """
        vpc_result = None
        ec2 = boto3.client('ec2', region_name=self.parameters['region'])
        vpc_input = {'SubnetIds': [self.parameters['subnet_id']]}
        try:
            vpc_result = ec2.describe_subnets(**vpc_input)
        except ClientError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        return vpc_result['Subnets'][0]['VpcId']

    def create_cvo_aws(self):
        """ Create AWS CVO """
        if self.parameters.get('workspace_id') is None:
            response, msg = self.na_helper.get_tenant(self.rest_api, self.headers)
            if response is None:
                self.module.fail_json(msg)
            self.parameters['workspace_id'] = response
        if self.parameters.get('vpc_id') is None and self.parameters['is_ha'] is False:
            self.parameters['vpc_id'] = self.get_vpc()
        if self.parameters.get('nss_account') is None:
            if self.parameters.get('platform_serial_number') is not None:
                if not self.parameters['platform_serial_number'].startswith('Eval-') and self.parameters['license_type'] == 'cot-premium-byol':
                    response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                    if response is None:
                        self.module.fail_json(msg)
                    self.parameters['nss_account'] = response
            elif self.parameters.get('platform_serial_number_node1') is not None and self.parameters.get('platform_serial_number_node2') is not None:
                if not self.parameters['platform_serial_number_node1'].startswith('Eval-') and (not self.parameters['platform_serial_number_node2'].startswith('Eval-')) and (self.parameters['license_type'] == 'ha-cot-premium-byol'):
                    response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                    if response is None:
                        self.module.fail_json(msg)
                    self.parameters['nss_account'] = response
        json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'tenantId': self.parameters['workspace_id'], 'vpcId': self.parameters['vpc_id'], 'dataEncryptionType': self.parameters['data_encryption_type'], 'ebsVolumeSize': {'size': self.parameters['ebs_volume_size'], 'unit': self.parameters['ebs_volume_size_unit']}, 'ebsVolumeType': self.parameters['ebs_volume_type'], 'svmPassword': self.parameters['svm_password'], 'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs'], 'enableCompliance': self.parameters['enable_compliance'], 'enableMonitoring': self.parameters['enable_monitoring'], 'optimizedNetworkUtilization': self.parameters['optimized_network_utilization'], 'vsaMetadata': {'ontapVersion': self.parameters['ontap_version'], 'licenseType': self.parameters['license_type'], 'useLatestVersion': self.parameters['use_latest_version'], 'instanceType': self.parameters['instance_type']}}
        if self.parameters['capacity_tier'] == 'S3':
            json.update({'capacityTier': self.parameters['capacity_tier'], 'tierLevel': self.parameters['tier_level']})
        if not self.parameters['license_type'].endswith('capacity-paygo'):
            json['vsaMetadata'].update({'capacityPackageName': ''})
        if self.parameters.get('platform_serial_number') is not None:
            json['vsaMetadata'].update({'platformSerialNumber': self.parameters['platform_serial_number']})
        if self.parameters.get('provided_license') is not None:
            json['vsaMetadata'].update({'providedLicense': self.parameters['provided_license']})
        if self.parameters.get('capacity_package_name') is not None:
            json['vsaMetadata'].update({'capacityPackageName': self.parameters['capacity_package_name']})
        if self.parameters.get('writing_speed_state') is not None:
            json.update({'writingSpeedState': self.parameters['writing_speed_state'].upper()})
        if self.parameters.get('iops') is not None:
            json.update({'iops': self.parameters['iops']})
        if self.parameters.get('throughput') is not None:
            json.update({'throughput': self.parameters['throughput']})
        if self.parameters.get('cluster_key_pair_name') is not None:
            json.update({'clusterKeyPairName': self.parameters['cluster_key_pair_name']})
        if self.parameters.get('instance_tenancy') is not None:
            json.update({'instanceTenancy': self.parameters['instance_tenancy']})
        if self.parameters.get('instance_profile_name') is not None:
            json.update({'instanceProfileName': self.parameters['instance_profile_name']})
        if self.parameters.get('security_group_id') is not None:
            json.update({'securityGroupId': self.parameters['security_group_id']})
        if self.parameters.get('cloud_provider_account') is not None:
            json.update({'cloudProviderAccount': self.parameters['cloud_provider_account']})
        if self.parameters.get('backup_volumes_to_cbs') is not None:
            json.update({'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs']})
        if self.parameters.get('svm_name') is not None:
            json.update({'svmName': self.parameters['svm_name']})
        if self.parameters['data_encryption_type'] == 'AWS':
            if self.parameters.get('kms_key_id') is not None:
                json.update({'awsEncryptionParameters': {'kmsKeyId': self.parameters['kms_key_id']}})
            if self.parameters.get('kms_key_arn') is not None:
                json.update({'awsEncryptionParameters': {'kmsKeyArn': self.parameters['kms_key_arn']}})
        if self.parameters.get('aws_tag') is not None:
            tags = []
            for each_tag in self.parameters['aws_tag']:
                tag = {'tagKey': each_tag['tag_key'], 'tagValue': each_tag['tag_value']}
                tags.append(tag)
            json.update({'awsTags': tags})
        if self.parameters['is_ha'] is True:
            ha_params = dict({'mediatorAssignPublicIP': self.parameters['mediator_assign_public_ip']})
            if self.parameters.get('failover_mode'):
                ha_params['failoverMode'] = self.parameters['failover_mode']
            if self.parameters.get('node1_subnet_id'):
                ha_params['node1SubnetId'] = self.parameters['node1_subnet_id']
            if self.parameters.get('node2_subnet_id'):
                ha_params['node2SubnetId'] = self.parameters['node2_subnet_id']
            if self.parameters.get('mediator_subnet_id'):
                ha_params['mediatorSubnetId'] = self.parameters['mediator_subnet_id']
            if self.parameters.get('mediator_key_pair_name'):
                ha_params['mediatorKeyPairName'] = self.parameters['mediator_key_pair_name']
            if self.parameters.get('cluster_floating_ip'):
                ha_params['clusterFloatingIP'] = self.parameters['cluster_floating_ip']
            if self.parameters.get('data_floating_ip'):
                ha_params['dataFloatingIP'] = self.parameters['data_floating_ip']
            if self.parameters.get('data_floating_ip2'):
                ha_params['dataFloatingIP2'] = self.parameters['data_floating_ip2']
            if self.parameters.get('svm_floating_ip'):
                ha_params['svmFloatingIP'] = self.parameters['svm_floating_ip']
            if self.parameters.get('route_table_ids'):
                ha_params['routeTableIds'] = self.parameters['route_table_ids']
            if self.parameters.get('platform_serial_number_node1'):
                ha_params['platformSerialNumberNode1'] = self.parameters['platform_serial_number_node1']
            if self.parameters.get('platform_serial_number_node2'):
                ha_params['platformSerialNumberNode2'] = self.parameters['platform_serial_number_node2']
            json['haParams'] = ha_params
        else:
            json['subnetId'] = self.parameters['subnet_id']
        api_url = '%s/working-environments' % self.rest_api.api_root_path
        response, error, on_cloud_request_id = self.rest_api.post(api_url, json, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on creating cvo aws: %s, %s' % (str(error), str(response)))
        working_environment_id = response['publicId']
        wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
        err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'create', 60, 60)
        if err is not None:
            self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating CVO AWS: %s' % str(err))
        return working_environment_id

    def update_cvo_aws(self, working_environment_id, modify):
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
            if item == 'aws_tag':
                tag_list = None
                if 'aws_tag' in self.parameters:
                    tag_list = self.parameters['aws_tag']
                response, error = self.na_helper.update_cvo_tags(base_url, self.rest_api, self.headers, 'aws_tag', tag_list)
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

    def delete_cvo_aws(self, we_id):
        """
        Delete AWS CVO
        """
        api_url = '%s/working-environments/%s' % (self.rest_api.api_root_path, we_id)
        response, error, on_cloud_request_id = self.rest_api.delete(api_url, None, header=self.headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on deleting cvo aws: %s, %s' % (str(error), str(response)))
        wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
        err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'delete', 40, 60)
        if err is not None:
            self.module.fail_json(msg='Error: unexpected response wait_on_completion for deleting CVO AWS: %s' % str(err))

    def validate_cvo_params(self):
        if self.parameters['use_latest_version'] is True and self.parameters['ontap_version'] != 'latest':
            self.module.fail_json(msg='ontap_version parameter not required when having use_latest_version as true')
        if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'ha-cot-premium-byol':
            if self.parameters.get('platform_serial_number_node1') is None or self.parameters.get('platform_serial_number_node2') is None:
                self.module.fail_json(msg='both platform_serial_number_node1 and platform_serial_number_node2 parameters are requiredwhen having ha type as true and license_type as ha-cot-premium-byol')
        if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'capacity-paygo':
            self.parameters['license_type'] = 'ha-capacity-paygo'

    def apply(self):
        """
        Apply action to the Cloud Manager CVO for AWS
        :return: None
        """
        working_environment_id = None
        modify = None
        current, dummy = self.na_helper.get_working_environment_details_by_name(self.rest_api, self.headers, self.parameters['name'], 'aws')
        if current:
            self.parameters['working_environment_id'] = current['publicId']
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if current and self.parameters['state'] != 'absent':
            self.validate_cvo_params()
            working_environment_id = current['publicId']
            modify, error = self.na_helper.is_cvo_update_needed(self.rest_api, self.headers, self.parameters, self.changeable_params, 'aws')
            if error is not None:
                self.module.fail_json(changed=False, msg=error)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.validate_cvo_params()
                working_environment_id = self.create_cvo_aws()
            elif cd_action == 'delete':
                self.delete_cvo_aws(current['publicId'])
            else:
                self.update_cvo_aws(current['publicId'], modify)
        self.module.exit_json(changed=self.na_helper.changed, working_environment_id=working_environment_id)