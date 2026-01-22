from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _override_vlans_if_all_specified(self):
    """Overrides any specified VLANs if "all" VLANs are specified

        The special setting "all VLANs" in a BIG-IP requires that no other VLANs
        be specified. If you specify any number of VLANs, AND include the "all"
        VLAN, this method will erase all of the other VLANs and only return the
        "all" VLAN.
        """
    all_vlans = ['/common/all', 'all']
    if self.want.enabled_vlans is not None:
        if any((x for x in self.want.enabled_vlans if x.lower() in all_vlans)):
            self.want.update(dict(enabled_vlans=[], vlans_disabled=True, vlans_enabled=False))