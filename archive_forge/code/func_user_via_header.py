from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def user_via_header(self):
    if self.want.user_via_header is None:
        return None
    if self.want.user_via_header == '':
        if self.have.user_via_header in [None, 'none']:
            return None
    if self.want.user_via_header != self.have.user_via_header:
        return self.want.user_via_header