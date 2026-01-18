from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def tsig_server_key(self):
    if self.want.tsig_server_key is None:
        return None
    if self.want.tsig_server_key == '' and self.have.tsig_server_key is None:
        return None
    if self.want.tsig_server_key != self.have.tsig_server_key:
        return self.want.tsig_server_key