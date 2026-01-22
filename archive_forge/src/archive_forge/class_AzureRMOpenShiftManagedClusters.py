from __future__ import absolute_import, division, print_function
import time
import json
import random
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMOpenShiftManagedClusters(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', updatable=False, disposition='resourceGroupName', required=True), name=dict(type='str', updatable=False, disposition='resourceName', required=True), location=dict(type='str', updatable=False, required=True, disposition='/'), cluster_profile=dict(type='dict', disposition='/properties/clusterProfile', default=dict(), options=dict(pull_secret=dict(type='str', no_log=True, updatable=False, disposition='pullSecret', purgeIfNone=True), cluster_resource_group_id=dict(type='str', updatable=False, disposition='resourceGroupId', purgeIfNone=True), domain=dict(type='str', updatable=False, disposition='domain', purgeIfNone=True), version=dict(type='str', updatable=False, disposition='version', purgeIfNone=True))), service_principal_profile=dict(type='dict', disposition='/properties/servicePrincipalProfile', options=dict(client_id=dict(type='str', updatable=False, disposition='clientId', required=True), client_secret=dict(type='str', no_log=True, updatable=False, disposition='clientSecret', required=True))), network_profile=dict(type='dict', disposition='/properties/networkProfile', options=dict(pod_cidr=dict(type='str', updatable=False, disposition='podCidr'), service_cidr=dict(type='str', updatable=False, disposition='serviceCidr')), default=dict(pod_cidr='10.128.0.0/14', service_cidr='172.30.0.0/16')), master_profile=dict(type='dict', disposition='/properties/masterProfile', options=dict(vm_size=dict(type='str', updatable=False, disposition='vmSize', choices=['Standard_D8s_v3', 'Standard_D16s_v3', 'Standard_D32s_v3'], purgeIfNone=True), subnet_id=dict(type='str', updatable=False, disposition='subnetId', required=True))), worker_profiles=dict(type='list', elements='dict', disposition='/properties/workerProfiles', options=dict(name=dict(type='str', disposition='name', updatable=False, required=True, choices=['worker']), count=dict(type='int', disposition='count', updatable=False, purgeIfNone=True), vm_size=dict(type='str', disposition='vmSize', updatable=False, choices=['Standard_D4s_v3', 'Standard_D8s_v3'], purgeIfNone=True), subnet_id=dict(type='str', disposition='subnetId', updatable=False, required=True), disk_size=dict(type='int', disposition='diskSizeGB', updatable=False, purgeIfNone=True))), api_server_profile=dict(type='dict', disposition='/properties/apiserverProfile', options=dict(visibility=dict(type='str', disposition='visibility', choices=['Public', 'Private'], default='Public'), url=dict(type='str', disposition='*', updatable=False), ip=dict(type='str', disposition='*', updatable=False))), ingress_profiles=dict(type='list', elements='dict', disposition='/properties/ingressProfiles', options=dict(name=dict(type='str', disposition='name', updatable=False, choices=['default'], default='default'), visibility=dict(type='str', disposition='visibility', updatable=False, choices=['Public', 'Private'], default='Public'), ip=dict(type='str', disposition='*', updatable=False))), provisioning_state=dict(type='str', disposition='/properties/provisioningState'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.header_parameters = {}
        self.query_parameters['api-version'] = '2020-04-30'
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMOpenShiftManagedClusters, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.RedHatOpenShift' + '/openShiftClusters' + '/{{ open_shift_managed_cluster_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ open_shift_managed_cluster_name }}', self.name)
        old_response = self.get_resource()
        if not old_response:
            self.log("OpenShiftManagedCluster instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('OpenShiftManagedCluster instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.fail("module doesn't support cluster update yet")
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the OpenShiftManagedCluster instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('OpenShiftManagedCluster instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
            while self.get_resource():
                time.sleep(20)
        else:
            self.log('OpenShiftManagedCluster instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['name'] = response['name']
            self.results['type'] = response['type']
            self.results['location'] = response['location']
            self.results['properties'] = response['properties']
        return self.results

    def create_update_resource(self):
        if self.to_do == Actions.Create:
            required_profile_for_creation = ['workerProfiles', 'clusterProfile', 'servicePrincipalProfile', 'masterProfile']
            if 'properties' not in self.body:
                self.fail('{0} are required for creating a openshift cluster'.format('[worker_profile, cluster_profile, service_principal_profile, master_profile]'))
            for profile in required_profile_for_creation:
                if profile not in self.body['properties']:
                    self.fail('{0} is required for creating a openshift cluster'.format(profile))
            self.set_default()
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error attempting to create the OpenShiftManagedCluster instance.')
            self.fail('Error creating the OpenShiftManagedCluster instance: {0}\n{1}'.format(str(self.body), str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
            pass
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete the OpenShiftManagedCluster instance.')
            self.fail('Error deleting the OpenShiftManagedCluster instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            found = True
            response = json.loads(response.body())
            found = True
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the OpenShiftManagedCluster instance.')
        if found is True:
            return response
        return False

    def random_id(self):
        random_id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz')) + ''.join((random.choice('abcdefghijklmnopqrstuvwxyz1234567890') for key in range(7)))
        return random_id

    def set_default(self):
        if 'apiserverProfile' not in self.body['properties']:
            api_profile = dict(visibility='Public')
            self.body['properties']['apiserverProfile'] = api_profile
        if 'ingressProfiles' not in self.body['properties']:
            ingress_profile = dict(visibility='Public', name='default')
            self.body['properties']['ingressProfiles'] = [ingress_profile]
        else:
            for profile in self.body['properties']['ingressProfiles']:
                profile['name'] = 'default'
        if 'name' not in self.body['properties']['workerProfiles'][0]:
            self.body['properties']['workerProfiles'][0]['name'] = 'worker'
        if 'vmSize' not in self.body['properties']['workerProfiles'][0]:
            self.body['properties']['workerProfiles'][0]['vmSize'] = 'Standard_D4s_v3'
        if 'diskSizeGB' not in self.body['properties']['workerProfiles'][0]:
            self.body['properties']['workerProfiles'][0]['diskSizeGB'] = 128
        if 'vmSize' not in self.body['properties']['masterProfile']:
            self.body['properties']['masterProfile']['vmSize'] = 'Standard_D8s_v3'
        if 'pullSecret' not in self.body['properties']['clusterProfile']:
            self.body['properties']['clusterProfile']['pullSecret'] = ''
        if 'resourceGroupId' not in self.body['properties']['clusterProfile']:
            resourcegroup_id = '/subscriptions/' + self.subscription_id + '/resourceGroups/' + self.name + '-cluster'
            self.body['properties']['clusterProfile']['resourceGroupId'] = resourcegroup_id
        if 'domain' not in self.body['properties']['clusterProfile'] or not self.body['properties']['clusterProfile']['domain']:
            self.body['properties']['clusterProfile']['domain'] = self.random_id()