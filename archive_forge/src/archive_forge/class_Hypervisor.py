import warnings
from openstack import exceptions
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
class Hypervisor(resource.Resource):
    resource_key = 'hypervisor'
    resources_key = 'hypervisors'
    base_path = '/os-hypervisors'
    allow_fetch = True
    allow_list = True
    _query_mapping = resource.QueryParameters('hypervisor_hostname_pattern', 'with_servers')
    _max_microversion = '2.88'
    cpu_info = resource.Body('cpu_info')
    host_ip = resource.Body('host_ip')
    hypervisor_type = resource.Body('hypervisor_type')
    hypervisor_version = resource.Body('hypervisor_version')
    name = resource.Body('hypervisor_hostname')
    service_details = resource.Body('service', type=dict)
    servers = resource.Body('servers', type=list, list_type=dict)
    state = resource.Body('state')
    status = resource.Body('status')
    uptime = resource.Body('uptime')
    current_workload = resource.Body('current_workload', deprecated=True)
    disk_available = resource.Body('disk_available_least', deprecated=True)
    local_disk_used = resource.Body('local_gb_used', deprecated=True)
    local_disk_size = resource.Body('local_gb', deprecated=True)
    local_disk_free = resource.Body('free_disk_gb', deprecated=True)
    memory_used = resource.Body('memory_mb_used', deprecated=True)
    memory_size = resource.Body('memory_mb', deprecated=True)
    memory_free = resource.Body('free_ram_mb', deprecated=True)
    running_vms = resource.Body('running_vms', deprecated=True)
    vcpus_used = resource.Body('vcpus_used', deprecated=True)
    vcpus = resource.Body('vcpus', deprecated=True)

    def get_uptime(self, session):
        """Get uptime information for the hypervisor

        Updates uptime attribute of the hypervisor object
        """
        warnings.warn('This call is deprecated and is only available until Nova 2.88', os_warnings.LegacyAPIWarning)
        if utils.supports_microversion(session, '2.88'):
            raise exceptions.SDKException('Hypervisor.get_uptime is not supported anymore')
        url = utils.urljoin(self.base_path, self.id, 'uptime')
        microversion = self._get_microversion(session, action='fetch')
        response = session.get(url, microversion=microversion)
        self._translate_response(response)
        return self