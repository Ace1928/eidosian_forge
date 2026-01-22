from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _munge_creation_state_for_device(self):
    if self.want.state in ['present', 'enabled']:
        self.want.update(dict(session='user-enabled', state='user-up'))
    elif self.want.state in 'disabled':
        self.want.update(dict(session='user-disabled', state='user-up'))
    else:
        self.want.update(dict(session='user-disabled', state='user-down', is_offline=True))