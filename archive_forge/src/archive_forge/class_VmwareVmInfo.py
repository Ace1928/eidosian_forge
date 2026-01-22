from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareVmInfo(PyVmomi):

    def __init__(self, module):
        super(VmwareVmInfo, self).__init__(module)
        if self.module.params.get('show_tag'):
            self.vmware_client = VmwareRestClient(self.module)

    def get_tag_info(self, vm_dynamic_obj):
        return self.vmware_client.get_tags_for_vm(vm_mid=vm_dynamic_obj._moId)

    def get_vm_attributes(self, vm):
        return dict(((x.name, v.value) for x in self.custom_field_mgr for v in vm.customValue if x.key == v.key))

    def get_virtual_machines(self):
        """
        Get one/all virtual machines and related configurations information.
        """
        folder = self.params.get('folder')
        folder_obj = None
        if folder:
            folder_obj = self.content.searchIndex.FindByInventoryPath(folder)
            if not folder_obj:
                self.module.fail_json(msg='Failed to find folder specified by %(folder)s' % self.params)
        vm_name = self.params.get('vm_name')
        if vm_name:
            virtual_machine = find_vm_by_name(self.content, vm_name=vm_name, folder=folder_obj)
            if not virtual_machine:
                self.module.fail_json(msg='Failed to find virtual machine %s' % vm_name)
            else:
                virtual_machines = [virtual_machine]
        else:
            virtual_machines = get_all_objs(self.content, [vim.VirtualMachine], folder=folder_obj)
        _virtual_machines = []
        for vm in virtual_machines:
            _ip_address = ''
            summary = vm.summary
            if summary.guest is not None:
                _ip_address = summary.guest.ipAddress
                if _ip_address is None:
                    _ip_address = ''
            _mac_address = []
            if self.module.params.get('show_mac_address'):
                all_devices = _get_vm_prop(vm, ('config', 'hardware', 'device'))
                if all_devices:
                    for dev in all_devices:
                        if isinstance(dev, vim.vm.device.VirtualEthernetCard):
                            _mac_address.append(dev.macAddress)
            net_dict = {}
            if self.module.params.get('show_net'):
                vmnet = _get_vm_prop(vm, ('guest', 'net'))
                if vmnet:
                    for device in vmnet:
                        net_dict[device.macAddress] = dict()
                        net_dict[device.macAddress]['ipv4'] = []
                        net_dict[device.macAddress]['ipv6'] = []
                        if device.ipConfig is not None:
                            for ip_addr in device.ipConfig.ipAddress:
                                if '::' in ip_addr.ipAddress:
                                    net_dict[device.macAddress]['ipv6'].append(ip_addr.ipAddress + '/' + str(ip_addr.prefixLength))
                                else:
                                    net_dict[device.macAddress]['ipv4'].append(ip_addr.ipAddress + '/' + str(ip_addr.prefixLength))
            esxi_hostname = None
            esxi_parent = None
            if self.module.params.get('show_esxi_hostname') or self.module.params.get('show_cluster'):
                if summary.runtime.host:
                    esxi_hostname = summary.runtime.host.summary.config.name
                    esxi_parent = summary.runtime.host.parent
            cluster_name = None
            if self.module.params.get('show_cluster'):
                if esxi_parent and isinstance(esxi_parent, vim.ClusterComputeResource):
                    cluster_name = summary.runtime.host.parent.name
            resource_pool = None
            if self.module.params.get('show_resource_pool'):
                if vm.resourcePool and vm.resourcePool != vm.resourcePool.owner.resourcePool:
                    resource_pool = vm.resourcePool.name
            vm_attributes = dict()
            if self.module.params.get('show_attribute'):
                vm_attributes = self.get_vm_attributes(vm)
            vm_tags = list()
            if self.module.params.get('show_tag'):
                vm_tags = self.get_tag_info(vm)
            allocated = {}
            if self.module.params.get('show_allocated'):
                storage_allocated = 0
                for device in vm.config.hardware.device:
                    if isinstance(device, vim.vm.device.VirtualDisk):
                        storage_allocated += device.capacityInBytes
                allocated = {'storage': storage_allocated, 'cpu': vm.config.hardware.numCPU, 'memory': vm.config.hardware.memoryMB}
            vm_folder = None
            if self.module.params.get('show_folder'):
                vm_folder = PyVmomi.get_vm_path(content=self.content, vm_name=vm)
            datacenter = None
            if self.module.params.get('show_datacenter'):
                datacenter = get_parent_datacenter(vm)
            datastore_url = list()
            if self.module.params.get('show_datastore'):
                datastore_attributes = ('name', 'url')
                vm_datastore_urls = _get_vm_prop(vm, ('config', 'datastoreUrl'))
                if vm_datastore_urls:
                    for entry in vm_datastore_urls:
                        datastore_url.append({key: getattr(entry, key) for key in dir(entry) if key in datastore_attributes})
            virtual_machine = {'guest_name': summary.config.name, 'guest_fullname': summary.config.guestFullName, 'power_state': summary.runtime.powerState, 'ip_address': _ip_address, 'mac_address': _mac_address, 'uuid': summary.config.uuid, 'instance_uuid': summary.config.instanceUuid, 'vm_network': net_dict, 'esxi_hostname': esxi_hostname, 'datacenter': None if datacenter is None else datacenter.name, 'cluster': cluster_name, 'resource_pool': resource_pool, 'attributes': vm_attributes, 'tags': vm_tags, 'folder': vm_folder, 'moid': vm._moId, 'datastore_url': datastore_url, 'allocated': allocated}
            vm_type = self.module.params.get('vm_type')
            is_template = _get_vm_prop(vm, ('config', 'template'))
            if vm_type == 'vm' and (not is_template):
                _virtual_machines.append(virtual_machine)
            elif vm_type == 'template' and is_template:
                _virtual_machines.append(virtual_machine)
            elif vm_type == 'all':
                _virtual_machines.append(virtual_machine)
        return _virtual_machines