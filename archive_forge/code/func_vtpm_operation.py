from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def vtpm_operation(self, vm_obj=None):
    results = {'failed': False, 'changed': False}
    if not self.is_vcenter():
        self.module.fail_json(msg='Please connect to vCenter Server to configure vTPM device of virtual machine.')
    self.vm = vm_obj
    if self.vm.runtime.powerState != vim.VirtualMachinePowerState.poweredOff:
        self.module.fail_json(msg="Please make sure VM is powered off before configuring vTPM device, current state is '%s'" % self.vm.runtime.powerState)
    for device in self.vm.config.hardware.device:
        if self.device_helper.is_tpm_device(device):
            self.vtpm_device = device
    if self.module.params['state'] == 'present':
        if self.module.check_mode:
            results['desired_operation'] = 'add vTPM'
        else:
            results['vtpm_operation'] = 'add vTPM'
        if self.vtpm_device:
            results['vtpm_info'] = self.get_vtpm_info(vtpm_device=self.vtpm_device)
            results['msg'] = 'vTPM device already exist on VM'
            self.module.exit_json(**results)
        else:
            if self.module.check_mode:
                results['changed'] = True
                self.module.exit_json(**results)
            vtpm_device_spec = self.device_helper.create_tpm()
    if self.module.params['state'] == 'absent':
        if self.module.check_mode:
            results['desired_operation'] = 'remove vTPM'
        else:
            results['vtpm_operation'] = 'remove vTPM'
        if self.vtpm_device is None:
            results['msg'] = 'No vTPM device found on VM'
            self.module.exit_json(**results)
        else:
            if self.module.check_mode:
                results['changed'] = True
                self.module.exit_json(**results)
            vtpm_device_spec = self.device_helper.remove_tpm(self.vtpm_device)
    self.config_spec.deviceChange.append(vtpm_device_spec)
    try:
        task = self.vm.ReconfigVM_Task(spec=self.config_spec)
        wait_for_task(task)
    except Exception as e:
        self.module.fail_json(msg="Failed to configure vTPM device on virtual machine due to '%s'" % to_native(e))
    if task.info.state == 'error':
        self.module.fail_json(msg='Failed to reconfigure VM with vTPM device', detail=task.info.error.msg)
    results['changed'] = True
    results['vtpm_info'] = self.get_vtpm_info(vm_obj=self.vm)
    self.module.exit_json(**results)