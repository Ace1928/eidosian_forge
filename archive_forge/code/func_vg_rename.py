from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
import random
def vg_rename(self, vg_data):
    msg = ''
    self.parameter_handling_while_renaming()
    old_vg_data = self.get_existing_vg(self.old_name)
    if not old_vg_data and (not vg_data):
        self.module.fail_json(msg="Volume group with old name {0} doesn't exist.".format(self.old_name))
    elif old_vg_data and vg_data:
        self.module.fail_json(msg='Volume group [{0}] already exists.'.format(self.name))
    elif not old_vg_data and vg_data:
        msg = 'Volume group with name [{0}] already exists.'.format(self.name)
    elif old_vg_data and (not vg_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chvolumegroup', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'Volume group [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg