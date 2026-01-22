from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class VmPoolsModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super(VmPoolsModule, self).__init__(*args, **kwargs)
        self._initialization = None

    def build_entity(self):
        vm = self.param('vm')
        return otypes.VmPool(id=self._module.params['id'], name=self._module.params['name'], description=self._module.params['description'], comment=self._module.params['comment'], cluster=otypes.Cluster(name=self._module.params['cluster']) if self._module.params['cluster'] else None, template=otypes.Template(name=self._module.params['template']) if self._module.params['template'] else None, max_user_vms=self._module.params['vm_per_user'], prestarted_vms=self._module.params['prestarted'], size=self._module.params['vm_count'], type=otypes.VmPoolType(self._module.params['type']) if self._module.params['type'] else None, vm=self.build_vm(vm) if self._module.params['vm'] else None)

    def build_vm(self, vm):
        return otypes.Vm(comment=vm.get('comment'), memory=convert_to_bytes(vm.get('memory')) if vm.get('memory') else None, memory_policy=otypes.MemoryPolicy(guaranteed=convert_to_bytes(vm.get('memory_guaranteed')), max=convert_to_bytes(vm.get('memory_max'))) if any((vm.get('memory_guaranteed'), vm.get('memory_max'))) else None, initialization=self.get_initialization(vm), display=otypes.Display(smartcard_enabled=vm.get('smartcard_enabled')) if vm.get('smartcard_enabled') is not None else None, sso=otypes.Sso(methods=[otypes.Method(id=otypes.SsoMethod.GUEST_AGENT)] if vm.get('sso') else []) if vm.get('sso') is not None else None, time_zone=otypes.TimeZone(name=vm.get('timezone')) if vm.get('timezone') else None)

    def get_initialization(self, vm):
        if self._initialization is not None:
            return self._initialization
        sysprep = vm.get('sysprep')
        cloud_init = vm.get('cloud_init')
        cloud_init_nics = vm.get('cloud_init_nics') or []
        if cloud_init is not None:
            cloud_init_nics.append(cloud_init)
        if cloud_init or cloud_init_nics:
            self._initialization = otypes.Initialization(nic_configurations=[otypes.NicConfiguration(boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol').lower()) if nic.get('nic_boot_protocol') else None, name=nic.pop('nic_name', None), on_boot=True, ip=otypes.Ip(address=nic.pop('nic_ip_address', None), netmask=nic.pop('nic_netmask', None), gateway=nic.pop('nic_gateway', None)) if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None else None) for nic in cloud_init_nics if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None or (nic.get('nic_boot_protocol') is not None)] if cloud_init_nics else None, **cloud_init)
        elif sysprep:
            self._initialization = otypes.Initialization(**sysprep)
        return self._initialization

    def get_vms(self, entity):
        vms = self._connection.system_service().vms_service().list()
        resp = []
        for vm in vms:
            if vm.vm_pool is not None and vm.vm_pool.id == entity.id:
                resp.append(vm)
        return resp

    def post_create(self, entity):
        vm_param = self.param('vm')
        if vm_param is not None and vm_param.get('nics') is not None:
            vms = self.get_vms(entity)
            for vm in vms:
                self.__attach_nics(vm, vm_param)

    def __attach_nics(self, entity, vm_param):
        vms_service = self._connection.system_service().vms_service()
        nics_service = vms_service.service(entity.id).nics_service()
        for nic in vm_param.get('nics'):
            if search_by_name(nics_service, nic.get('name')) is None:
                if not self._module.check_mode:
                    nics_service.add(otypes.Nic(name=nic.get('name'), interface=otypes.NicInterface(nic.get('interface', 'virtio')), vnic_profile=otypes.VnicProfile(id=self.__get_vnic_profile_id(nic)) if nic.get('profile_name') else None, mac=otypes.Mac(address=nic.get('mac_address')) if nic.get('mac_address') else None))
                self.changed = True

    def __get_vnic_profile_id(self, nic):
        """
        Return VNIC profile ID looked up by it's name, because there can be
        more VNIC profiles with same name, other criteria of filter is cluster.
        """
        vnics_service = self._connection.system_service().vnic_profiles_service()
        clusters_service = self._connection.system_service().clusters_service()
        cluster = search_by_name(clusters_service, self.param('cluster'))
        profiles = [profile for profile in vnics_service.list() if profile.name == nic.get('profile_name')]
        cluster_networks = [net.id for net in self._connection.follow_link(cluster.networks)]
        try:
            return next((profile.id for profile in profiles if profile.network.id in cluster_networks))
        except StopIteration:
            raise Exception("Profile '%s' was not found in cluster '%s'" % (nic.get('profile_name'), self.param('cluster')))

    def update_check(self, entity):
        return equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('cluster'), get_link_name(self._connection, entity.cluster)) and equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('comment'), entity.comment) and equal(self._module.params.get('vm_per_user'), entity.max_user_vms) and equal(self._module.params.get('prestarted'), entity.prestarted_vms) and equal(self._module.params.get('vm_count'), entity.size)