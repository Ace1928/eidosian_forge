from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_vm_by_id
class VmBootInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmBootInfoManager, self).__init__(module)
        self.name = self.params['name']
        self.uuid = self.params['uuid']
        self.moid = self.params['moid']
        self.use_instance_uuid = self.params['use_instance_uuid']
        self.vm = None

    def _get_vm(self):
        vms = []
        if self.uuid:
            if self.use_instance_uuid:
                vm_obj = find_vm_by_id(self.content, vm_id=self.uuid, vm_id_type='use_instance_uuid')
            else:
                vm_obj = find_vm_by_id(self.content, vm_id=self.uuid, vm_id_type='uuid')
            if vm_obj is None:
                self.module.fail_json(msg='Failed to find the virtual machine with UUID : %s' % self.uuid)
            vms = [vm_obj]
        elif self.name:
            objects = self.get_managed_objects_properties(vim_type=vim.VirtualMachine, properties=['name'])
            for temp_vm_object in objects:
                if temp_vm_object.obj.name == self.name:
                    vms.append(temp_vm_object.obj)
        elif self.moid:
            vm_obj = VmomiSupport.templateOf('VirtualMachine')(self.module.params['moid'], self.si._stub)
            if vm_obj:
                vms.append(vm_obj)
        if vms:
            if self.params.get('name_match') == 'first':
                self.vm = vms[0]
            elif self.params.get('name_match') == 'last':
                self.vm = vms[-1]
        else:
            self.module.fail_json(msg='Failed to find virtual machine using %s' % (self.name or self.uuid or self.moid))

    @staticmethod
    def humanize_boot_order(boot_order):
        results = []
        for device in boot_order:
            if isinstance(device, vim.vm.BootOptions.BootableCdromDevice):
                results.append('cdrom')
            elif isinstance(device, vim.vm.BootOptions.BootableDiskDevice):
                results.append('disk')
            elif isinstance(device, vim.vm.BootOptions.BootableEthernetDevice):
                results.append('ethernet')
            elif isinstance(device, vim.vm.BootOptions.BootableFloppyDevice):
                results.append('floppy')
        return results

    def ensure(self):
        self._get_vm()
        results = dict()
        if self.vm and self.vm.config:
            results = dict(current_boot_order=self.humanize_boot_order(self.vm.config.bootOptions.bootOrder), current_boot_delay=self.vm.config.bootOptions.bootDelay, current_enter_bios_setup=self.vm.config.bootOptions.enterBIOSSetup, current_boot_retry_enabled=self.vm.config.bootOptions.bootRetryEnabled, current_boot_retry_delay=self.vm.config.bootOptions.bootRetryDelay, current_boot_firmware=self.vm.config.firmware, current_secure_boot_enabled=self.vm.config.bootOptions.efiSecureBootEnabled)
        self.module.exit_json(changed=False, vm_boot_info=results)