from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def wait_for_vm_ip_address(module, vm_ref, timeout=300):
    """Waits for VM to acquire an IP address.

    Args:
        module: Reference to Ansible module object.
        vm_ref (str): XAPI reference to VM.
        timeout (int): timeout in seconds (default: 300).

    Returns:
        dict: VM guest metrics as retrieved by
        VM_guest_metrics.get_record() XAPI method with info
        on IP address acquired.
    """
    if not vm_ref or vm_ref == 'OpaqueRef:NULL':
        module.fail_json(msg='Cannot wait for VM IP address. Invalid VM reference supplied!')
    xapi_session = XAPI.connect(module)
    vm_guest_metrics = {}
    try:
        vm_power_state = xapi_to_module_vm_power_state(xapi_session.xenapi.VM.get_power_state(vm_ref).lower())
        if vm_power_state != 'poweredon':
            module.fail_json(msg="Cannot wait for VM IP address when VM is in state '%s'!" % vm_power_state)
        interval = 2
        if timeout == 0:
            time_left = 1
        else:
            time_left = timeout
        while time_left > 0:
            vm_guest_metrics_ref = xapi_session.xenapi.VM.get_guest_metrics(vm_ref)
            if vm_guest_metrics_ref != 'OpaqueRef:NULL':
                vm_guest_metrics = xapi_session.xenapi.VM_guest_metrics.get_record(vm_guest_metrics_ref)
                vm_ips = vm_guest_metrics['networks']
                if '0/ip' in vm_ips:
                    break
            time.sleep(interval)
            if timeout != 0:
                time_left -= interval
        else:
            module.fail_json(msg='Timed out waiting for VM IP address!')
    except XenAPI.Failure as f:
        module.fail_json(msg='XAPI ERROR: %s' % f.details)
    return vm_guest_metrics