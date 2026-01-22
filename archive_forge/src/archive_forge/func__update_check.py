from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _update_check(self, entity):

    def check_cpu_pinning():
        if self.param('cpu_pinning'):
            current = []
            if entity.cpu.cpu_tune:
                current = [(str(pin.cpu_set), int(pin.vcpu)) for pin in entity.cpu.cpu_tune.vcpu_pins]
            passed = [(str(pin['cpu']), int(pin['vcpu'])) for pin in self.param('cpu_pinning')]
            return sorted(current) == sorted(passed)
        return True

    def check_custom_properties():
        if self.param('custom_properties'):
            current = []
            if entity.custom_properties:
                current = [(cp.name, cp.regexp, str(cp.value)) for cp in entity.custom_properties]
            passed = [(cp.get('name'), cp.get('regexp'), str(cp.get('value'))) for cp in self.param('custom_properties') if cp]
            return sorted(current) == sorted(passed)
        return True

    def check_placement_policy():
        if self.param('placement_policy'):
            hosts = sorted(map(lambda host: self._connection.follow_link(host).name, entity.placement_policy.hosts if entity.placement_policy.hosts else []))
            if self.param('placement_policy_hosts'):
                return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal(sorted(self.param('placement_policy_hosts')), hosts)
            return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal([self.param('host')], hosts)
        return True

    def check_host():
        if self.param('host') is not None:
            return self.param('host') in [self._connection.follow_link(host).name for host in getattr(entity.placement_policy, 'hosts', None) or []]
        return True

    def check_custom_compatibility_version():
        if self.param('custom_compatibility_version') is not None:
            return self._get_minor(self.param('custom_compatibility_version')) == self._get_minor(entity.custom_compatibility_version) and self._get_major(self.param('custom_compatibility_version')) == self._get_major(entity.custom_compatibility_version)
        return True
    cpu_mode = getattr(entity.cpu, 'mode')
    vm_display = entity.display
    provided_vm_display = self.param('graphical_console') or dict()
    return check_cpu_pinning() and check_custom_properties() and check_host() and check_placement_policy() and check_custom_compatibility_version() and (not self.param('cloud_init_persist')) and (not self.param('kernel_params_persist')) and equal(self.param('cluster'), get_link_name(self._connection, entity.cluster)) and equal(convert_to_bytes(self.param('memory')), entity.memory) and equal(convert_to_bytes(self.param('memory_guaranteed')), getattr(entity.memory_policy, 'guaranteed', None)) and equal(convert_to_bytes(self.param('memory_max')), getattr(entity.memory_policy, 'max', None)) and equal(self.param('cpu_cores'), getattr(getattr(entity.cpu, 'topology', None), 'cores', None)) and equal(self.param('cpu_sockets'), getattr(getattr(entity.cpu, 'topology', None), 'sockets', None)) and equal(self.param('cpu_threads'), getattr(getattr(entity.cpu, 'topology', None), 'threads', None)) and equal(self.param('cpu_mode'), str(cpu_mode) if cpu_mode else None) and equal(self.param('type'), str(entity.type)) and equal(self.param('name'), str(entity.name)) and equal(self.param('operating_system'), str(getattr(entity.os, 'type', None))) and equal(self.param('boot_menu'), getattr(getattr(entity.bios, 'boot_menu', None), 'enabled', None)) and equal(self.param('bios_type'), getattr(getattr(entity.bios, 'type', None), 'value', None)) and equal(self.param('soundcard_enabled'), entity.soundcard_enabled) and equal(self.param('smartcard_enabled'), getattr(vm_display, 'smartcard_enabled', False)) and equal(self.param('io_threads'), getattr(entity.io, 'threads', None)) and equal(self.param('ballooning_enabled'), getattr(entity.memory_policy, 'ballooning', None)) and equal(self.param('serial_console'), getattr(entity.console, 'enabled', None)) and equal(self.param('usb_support'), getattr(entity.usb, 'enabled', None)) and equal(self.param('sso'), True if getattr(entity.sso, 'methods', False) else False) and equal(self.param('quota_id'), getattr(entity.quota, 'id', None)) and equal(self.param('high_availability'), getattr(entity.high_availability, 'enabled', None)) and equal(self.param('high_availability_priority'), getattr(entity.high_availability, 'priority', None)) and equal(self.param('lease'), get_link_name(self._connection, getattr(entity.lease, 'storage_domain', None))) and equal(self.param('stateless'), entity.stateless) and equal(self.param('cpu_shares'), entity.cpu_shares) and equal(self.param('delete_protected'), entity.delete_protected) and equal(self.param('custom_emulated_machine'), entity.custom_emulated_machine) and equal(self.param('use_latest_template_version'), entity.use_latest_template_version) and equal(self.param('boot_devices'), [str(dev) for dev in getattr(getattr(entity.os, 'boot', None), 'devices', [])]) and equal(self.param('instance_type'), get_link_name(self._connection, entity.instance_type), ignore_case=True) and equal(self.param('description'), entity.description) and equal(self.param('comment'), entity.comment) and equal(self.param('timezone'), getattr(entity.time_zone, 'name', None)) and equal(self.param('serial_policy'), str(getattr(entity.serial_number, 'policy', None))) and equal(self.param('serial_policy_value'), getattr(entity.serial_number, 'value', None)) and equal(self.param('numa_tune_mode'), str(entity.numa_tune_mode)) and equal(self.param('storage_error_resume_behaviour'), str(entity.storage_error_resume_behaviour)) and equal(self.param('virtio_scsi_enabled'), getattr(entity.virtio_scsi, 'enabled', None)) and equal(self.param('multi_queues_enabled'), entity.multi_queues_enabled) and equal(self.param('virtio_scsi_multi_queues'), entity.virtio_scsi_multi_queues) and equal(self.param('tpm_enabled'), entity.tpm_enabled) and equal(self.param('rng_device'), str(entity.rng_device.source) if entity.rng_device else None) and equal(provided_vm_display.get('monitors'), getattr(vm_display, 'monitors', None)) and equal(provided_vm_display.get('copy_paste_enabled'), getattr(vm_display, 'copy_paste_enabled', None)) and equal(provided_vm_display.get('file_transfer_enabled'), getattr(vm_display, 'file_transfer_enabled', None)) and equal(provided_vm_display.get('keyboard_layout'), getattr(vm_display, 'keyboard_layout', None)) and equal(provided_vm_display.get('disconnect_action'), getattr(vm_display, 'disconnect_action', None), ignore_case=True)