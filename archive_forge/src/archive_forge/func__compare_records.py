from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def _compare_records(self):
    want = self.want.records
    have = self.have.records
    if want == [] and have is None:
        return None
    if want is None:
        return None
    w = []
    h = []
    for x in want:
        tmp = tuple(((str(k), str(v)) for k, v in iteritems(x)))
        w.append(tmp)
    for x in have:
        tmp = tuple(((str(k), str(v)) for k, v in iteritems(x)))
        h.append(tmp)
    if set(w) == set(h):
        return None
    else:
        return want