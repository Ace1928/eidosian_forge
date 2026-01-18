from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def trusted_responders(self):
    if self.want.trusted_responders is None:
        return None
    if self.want.trusted_responders == '' and self.have.trusted_responders is None:
        return None
    if self.want.trusted_responders != self.have.trusted_responders:
        return self.want.trusted_responders