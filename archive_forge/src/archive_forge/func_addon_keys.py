from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
@property
def addon_keys(self):
    if self._values['license_key'] is None:
        return None
    if self._values['addon_keys'] is None or is_empty_list(self._values['addon_keys']):
        return None
    result = ','.join(self._values['addon_keys'])
    return result