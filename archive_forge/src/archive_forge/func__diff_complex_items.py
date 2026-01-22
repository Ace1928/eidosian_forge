from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _diff_complex_items(self, want, have):
    if want == [] and have is None:
        return None
    if want is None:
        return None
    w = self.to_tuple(want)
    h = self.to_tuple(have)
    if set(w).issubset(set(h)):
        return None
    else:
        return want