from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM

        Ensure VM storage profile policies.

        :param vm_obj: VMware VM object.
        :type vm_obj: VirtualMachine
        :exits: self.module.exit_json on success, else self.module.fail_json.
        