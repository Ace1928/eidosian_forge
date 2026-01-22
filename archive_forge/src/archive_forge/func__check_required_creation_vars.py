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
def _check_required_creation_vars(self):
    if self.want.address is None and self.want.fqdn is None:
        raise F5ModuleError("At least one of 'address' or 'fqdn' is required when creating a node")
    elif self.want.address is not None and self.want.fqdn is not None:
        raise F5ModuleError("Only one of 'address' or 'fqdn' can be provided when creating a node")
    elif self.want.fqdn is not None:
        self.want.update(dict(address='any6'))