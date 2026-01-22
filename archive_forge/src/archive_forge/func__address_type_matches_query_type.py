from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _address_type_matches_query_type(self, type, validator):
    if self.want.query_type == type and self.have.query_type == type:
        if self.want.receive is not None and validator(self.want.receive):
            return True
        if self.have.receive is not None and validator(self.have.receive):
            return True