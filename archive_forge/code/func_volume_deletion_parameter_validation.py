from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def volume_deletion_parameter_validation(self):
    if self.old_name:
        self.module.fail_json(msg='Parameter [old_name] is not supported during volume deletion.')