from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackServiceOffering(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackServiceOffering, self).__init__(module)
        self.returns = {'cpunumber': 'cpu_number', 'cpuspeed': 'cpu_speed', 'deploymentplanner': 'deployment_planner', 'diskBytesReadRate': 'disk_bytes_read_rate', 'diskBytesWriteRate': 'disk_bytes_write_rate', 'diskIopsReadRate': 'disk_iops_read_rate', 'diskIopsWriteRate': 'disk_iops_write_rate', 'maxiops': 'disk_iops_max', 'miniops': 'disk_iops_min', 'hypervisorsnapshotreserve': 'hypervisor_snapshot_reserve', 'iscustomized': 'is_customized', 'iscustomizediops': 'is_iops_customized', 'issystem': 'is_system', 'isvolatile': 'is_volatile', 'limitcpuuse': 'limit_cpu_usage', 'memory': 'memory', 'networkrate': 'network_rate', 'offerha': 'offer_ha', 'provisioningtype': 'provisioning_type', 'serviceofferingdetails': 'service_offering_details', 'storagetype': 'storage_type', 'systemvmtype': 'system_vm_type', 'tags': 'storage_tags'}

    def get_service_offering(self):
        args = {'name': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'issystem': self.module.params.get('is_system'), 'systemvmtype': self.module.params.get('system_vm_type')}
        service_offerings = self.query_api('listServiceOfferings', **args)
        if service_offerings:
            return service_offerings['serviceoffering'][0]

    def present_service_offering(self):
        service_offering = self.get_service_offering()
        if not service_offering:
            service_offering = self._create_offering(service_offering)
        else:
            service_offering = self._update_offering(service_offering)
        return service_offering

    def absent_service_offering(self):
        service_offering = self.get_service_offering()
        if service_offering:
            self.result['changed'] = True
            if not self.module.check_mode:
                args = {'id': service_offering['id']}
                self.query_api('deleteServiceOffering', **args)
        return service_offering

    def _create_offering(self, service_offering):
        self.result['changed'] = True
        system_vm_type = self.module.params.get('system_vm_type')
        is_system = self.module.params.get('is_system')
        required_params = []
        if is_system and (not system_vm_type):
            required_params.append('system_vm_type')
        self.module.fail_on_missing_params(required_params=required_params)
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'bytesreadrate': self.module.params.get('disk_bytes_read_rate'), 'byteswriterate': self.module.params.get('disk_bytes_write_rate'), 'cpunumber': self.module.params.get('cpu_number'), 'cpuspeed': self.module.params.get('cpu_speed'), 'customizediops': self.module.params.get('is_iops_customized'), 'deploymentplanner': self.module.params.get('deployment_planner'), 'domainid': self.get_domain(key='id'), 'hosttags': self.module.params.get('host_tags'), 'hypervisorsnapshotreserve': self.module.params.get('hypervisor_snapshot_reserve'), 'iopsreadrate': self.module.params.get('disk_iops_read_rate'), 'iopswriterate': self.module.params.get('disk_iops_write_rate'), 'maxiops': self.module.params.get('disk_iops_max'), 'miniops': self.module.params.get('disk_iops_min'), 'issystem': is_system, 'isvolatile': self.module.params.get('is_volatile'), 'memory': self.module.params.get('memory'), 'networkrate': self.module.params.get('network_rate'), 'offerha': self.module.params.get('offer_ha'), 'provisioningtype': self.module.params.get('provisioning_type'), 'serviceofferingdetails': self.module.params.get('service_offering_details'), 'storagetype': self.module.params.get('storage_type'), 'systemvmtype': system_vm_type, 'tags': self.module.params.get('storage_tags'), 'limitcpuuse': self.module.params.get('limit_cpu_usage'), 'customized': self.module.params.get('is_customized')}
        if not self.module.check_mode:
            res = self.query_api('createServiceOffering', **args)
            service_offering = res['serviceoffering']
        return service_offering

    def _update_offering(self, service_offering):
        args = {'id': service_offering['id'], 'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name')}
        if self.has_changed(args, service_offering):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateServiceOffering', **args)
                service_offering = res['serviceoffering']
        return service_offering

    def get_result(self, resource):
        super(AnsibleCloudStackServiceOffering, self).get_result(resource)
        if resource:
            if 'hosttags' in resource:
                self.result['host_tags'] = resource['hosttags'].split(',') or [resource['hosttags']]
            if 'tags' in resource:
                self.result['storage_tags'] = resource['tags'].split(',') or [resource['tags']]
            if 'tags' in self.result:
                del self.result['tags']
        return self.result