from __future__ import (absolute_import, division, print_function)
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, gather_vm_facts, vmware_argument_spec
def wait_for_tools(self, vm, timeout):
    tools_running = False
    vm_facts = {}
    start_at = datetime.datetime.now()
    while start_at + timeout > datetime.datetime.now():
        newvm = self.get_vm()
        vm_facts = self.gather_facts(newvm)
        if vm_facts['guest_tools_status'] == 'guestToolsRunning':
            return {'changed': True, 'failed': False, 'instance': vm_facts}
        time.sleep(5)
    if not tools_running:
        return {'failed': True, 'msg': 'VMware tools either not present or not running after {0} seconds'.format(timeout.total_seconds())}