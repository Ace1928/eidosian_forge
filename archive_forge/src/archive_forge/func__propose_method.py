from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _propose_method(self, default_method):
    if 'proposed_method' in self.module.params and self.module.params['proposed_method']:
        return self.module.params['proposed_method']
    return default_method