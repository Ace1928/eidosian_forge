from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def wait_for_sp_reboot(self):
    for dummy in range(20):
        time.sleep(15)
        state = self.get_sp_state()
        if state != 'rebooting':
            break
    else:
        self.module.warn('node did not finish up booting in 5 minutes!')