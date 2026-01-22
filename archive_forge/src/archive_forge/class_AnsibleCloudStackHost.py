from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackHost(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackHost, self).__init__(module)
        self.returns = {'averageload': 'average_load', 'capabilities': 'capabilities', 'clustername': 'cluster', 'clustertype': 'cluster_type', 'cpuallocated': 'cpu_allocated', 'cpunumber': 'cpu_number', 'cpusockets': 'cpu_sockets', 'cpuspeed': 'cpu_speed', 'cpuused': 'cpu_used', 'cpuwithoverprovisioning': 'cpu_with_overprovisioning', 'disconnected': 'disconnected', 'details': 'details', 'disksizeallocated': 'disk_size_allocated', 'disksizetotal': 'disk_size_total', 'events': 'events', 'hahost': 'ha_host', 'hasenoughcapacity': 'has_enough_capacity', 'hypervisor': 'hypervisor', 'hypervisorversion': 'hypervisor_version', 'ipaddress': 'ip_address', 'islocalstorageactive': 'is_local_storage_active', 'lastpinged': 'last_pinged', 'managementserverid': 'management_server_id', 'memoryallocated': 'memory_allocated', 'memorytotal': 'memory_total', 'memoryused': 'memory_used', 'networkkbsread': 'network_kbs_read', 'networkkbswrite': 'network_kbs_write', 'oscategoryname': 'os_category', 'outofbandmanagement': 'out_of_band_management', 'podname': 'pod', 'removed': 'removed', 'resourcestate': 'resource_state', 'suitableformigration': 'suitable_for_migration', 'type': 'host_type', 'version': 'host_version', 'gpugroup': 'gpu_group'}
        self.allocation_states_for_update = {'enabled': 'Enable', 'disabled': 'Disable'}
        self.host = None

    def get_cluster(self, key=None):
        cluster_name = self.module.params.get('cluster')
        if not cluster_name:
            return None
        args = {'name': cluster_name, 'zoneid': self.get_zone(key='id')}
        clusters = self.query_api('listClusters', **args)
        if clusters:
            return self._get_by_key(key, clusters['cluster'][0])
        self.module.fail_json(msg='Cluster %s not found' % cluster_name)

    def get_host_tags(self):
        host_tags = self.module.params.get('host_tags')
        if host_tags is None:
            return None
        return ','.join(host_tags)

    def get_host(self, refresh=False):
        if self.host is not None and (not refresh):
            return self.host
        name = self.module.params.get('name')
        args = {'zoneid': self.get_zone(key='id'), 'fetch_list': True}
        res = self.query_api('listHosts', **args)
        if res:
            for h in res:
                if name in [h['ipaddress'], h['name']]:
                    self.host = h
        return self.host

    def _handle_allocation_state(self, host):
        allocation_state = self.module.params.get('allocation_state')
        if not allocation_state:
            return host
        host = self._set_host_allocation_state(host)
        if host['allocationstate'].lower() == allocation_state and allocation_state == 'maintenance':
            return host
        elif allocation_state in list(self.allocation_states_for_update.keys()):
            host = self.disable_maintenance(host)
            host = self._update_host(host, self.allocation_states_for_update[allocation_state])
        elif allocation_state == 'maintenance':
            host = self._update_host(host, 'Enable')
            host = self.enable_maintenance(host)
        return host

    def _set_host_allocation_state(self, host):
        if host is None:
            host['allocationstate'] = 'Enable'
        elif host['resourcestate'].lower() in list(self.allocation_states_for_update.keys()):
            host['allocationstate'] = self.allocation_states_for_update[host['resourcestate'].lower()]
        else:
            host['allocationstate'] = host['resourcestate']
        return host

    def present_host(self):
        host = self.get_host()
        if not host:
            host = self._create_host(host)
        else:
            host = self._update_host(host)
        if host:
            host = self._handle_allocation_state(host)
        return host

    def _get_url(self):
        url = self.module.params.get('url')
        if url:
            return url
        else:
            return 'http://%s' % self.module.params.get('name')

    def _create_host(self, host):
        required_params = ['password', 'username', 'hypervisor', 'pod']
        self.module.fail_on_missing_params(required_params=required_params)
        self.result['changed'] = True
        args = {'hypervisor': self.module.params.get('hypervisor'), 'url': self._get_url(), 'username': self.module.params.get('username'), 'password': self.module.params.get('password'), 'podid': self.get_pod(key='id'), 'zoneid': self.get_zone(key='id'), 'clusterid': self.get_cluster(key='id'), 'hosttags': self.get_host_tags()}
        if not self.module.check_mode:
            host = self.query_api('addHost', **args)
            host = host['host'][0]
        return host

    def _update_host(self, host, allocation_state=None):
        args = {'id': host['id'], 'hosttags': self.get_host_tags(), 'allocationstate': allocation_state}
        if allocation_state is not None:
            host = self._set_host_allocation_state(host)
        if self.has_changed(args, host):
            self.result['changed'] = True
            if not self.module.check_mode:
                host = self.query_api('updateHost', **args)
                host = host['host']
        return host

    def absent_host(self):
        host = self.get_host()
        if host:
            self.result['changed'] = True
            args = {'id': host['id']}
            if not self.module.check_mode:
                res = self.enable_maintenance(host)
                if res:
                    res = self.query_api('deleteHost', **args)
        return host

    def enable_maintenance(self, host):
        if host['resourcestate'] not in ['PrepareForMaintenance', 'Maintenance']:
            self.result['changed'] = True
            args = {'id': host['id']}
            if not self.module.check_mode:
                res = self.query_api('prepareHostForMaintenance', **args)
                self.poll_job(res, 'host')
                host = self._poll_for_maintenance()
        return host

    def disable_maintenance(self, host):
        if host['resourcestate'] in ['PrepareForMaintenance', 'Maintenance']:
            self.result['changed'] = True
            args = {'id': host['id']}
            if not self.module.check_mode:
                res = self.query_api('cancelHostMaintenance', **args)
                host = self.poll_job(res, 'host')
        return host

    def _poll_for_maintenance(self):
        for i in range(0, 300):
            time.sleep(2)
            host = self.get_host(refresh=True)
            if not host:
                return None
            elif host['resourcestate'] != 'PrepareForMaintenance':
                return host
        self.fail_json(msg='Polling for maintenance timed out')

    def get_result(self, resource):
        super(AnsibleCloudStackHost, self).get_result(resource)
        if resource:
            self.result['allocation_state'] = resource['resourcestate'].lower()
            self.result['host_tags'] = resource['hosttags'].split(',') if resource.get('hosttags') else []
        return self.result