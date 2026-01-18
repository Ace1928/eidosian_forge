from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def version_is_less_than_3(self):
    version = self.module.params.get('version')
    if version == 'v3':
        return False
    else:
        return True