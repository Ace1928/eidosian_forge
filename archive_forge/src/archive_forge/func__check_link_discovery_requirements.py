from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _check_link_discovery_requirements(self):
    if self.want.link_discovery in ['enabled', 'enabled-no-delete'] and self.want.virtual_server_discovery == 'disabled':
        raise F5ModuleError('Virtual server discovery must be enabled if link discovery is enabled')