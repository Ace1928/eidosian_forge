from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _update_profile(self):
    if self.state == 'present':
        if self.old_state == 'absent':
            if self.new_name is None:
                self._create_profile()
            else:
                self.module.fail_json(msg='new_name must not be set when the profile does not exist and the state is present', changed=False)
        else:
            if self.new_name is not None and self.new_name != self.name:
                self._rename_profile()
            if self._needs_to_apply_profile_configs():
                self._apply_profile_configs()
    elif self.state == 'absent':
        if self.old_state == 'present':
            if self.new_name is None:
                self._delete_profile()
            else:
                self.module.fail_json(msg='new_name must not be set when the profile exists and the specified state is absent', changed=False)