from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def wait_for_customization(self, vm, timeout=3600, sleep=10):
    poll = int(timeout // sleep)
    thispoll = 0
    while thispoll <= poll:
        eventStarted = self.get_vm_events(vm, ['CustomizationStartedEvent'])
        if len(eventStarted):
            thispoll = 0
            while thispoll <= poll:
                eventsFinishedResult = self.get_vm_events(vm, ['CustomizationSucceeded', 'CustomizationFailed'])
                if len(eventsFinishedResult):
                    if not isinstance(eventsFinishedResult[0], vim.event.CustomizationSucceeded):
                        self.module.warn('Customization failed with error {%s}:{%s}' % (eventsFinishedResult[0]._wsdlName, eventsFinishedResult[0].fullFormattedMessage))
                        return False
                    else:
                        return True
                else:
                    time.sleep(sleep)
                    thispoll += 1
            if len(eventsFinishedResult) == 0:
                self.module.warn('Waiting for customization result event timed out.')
                return False
        else:
            time.sleep(sleep)
            thispoll += 1
    if len(eventStarted):
        self.module.warn('Waiting for customization result event timed out.')
    else:
        self.module.warn('Waiting for customization start event timed out.')
    return False