from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def wait_for_quiesced_status(self):
    for __ in range(5):
        time.sleep(5)
        sm_info = self.snapmirror_get()
        if sm_info['status'] == 'quiesced' or sm_info['mirror_state'] == 'paused':
            return
    self.module.fail_json(msg='Taking a long time to quiesce SnapMirror relationship, try again later')