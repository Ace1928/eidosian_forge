from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def volume_rename(self, volume_data):
    msg = None
    self.parameter_handling_while_renaming()
    old_volume_data = self.get_existing_volume(self.old_name)
    if not old_volume_data and (not volume_data):
        self.module.fail_json(msg='Volume [{0}] does not exists.'.format(self.old_name))
    elif old_volume_data and volume_data:
        self.module.fail_json(msg='Volume [{0}] already exists.'.format(self.name))
    elif not old_volume_data and volume_data:
        msg = 'Volume with name [{0}] already exists.'.format(self.name)
    elif old_volume_data and (not volume_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chvdisk', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'Volume [{0}] has been successfully rename to [{1}]'.format(self.old_name, self.name)
    return msg