from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_all_value(self):
    items = self._values['self_allow'].get('defaults')
    if isinstance(items, string_types):
        if items == 'none':
            return 'no'
        if items == 'all':
            return 'yes'