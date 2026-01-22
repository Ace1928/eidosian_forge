from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _process_with_mkey(self, mvalue):
    if self.module.params['state'] == 'present':
        rc, robject = self.get_object(mvalue)
        if rc == 0:
            if self._method_proposed() or self._update_required(robject):
                return self.update_object(mvalue)
            else:
                self.module.exit_json(message='Your FortiManager is up to date, no need to update. To force update, please add argument proposed_method:update')
        else:
            return self.create_object()
    elif self.module.params['state'] == 'absent':
        return self.delete_object(mvalue)