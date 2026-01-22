from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _method_proposed(self):
    return 'proposed_method' in self.module.params and self.module.params['proposed_method']