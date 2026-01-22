from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _read_ucs_files_from_output(self, output):
    search = re.compile('filename\\s+(.*)').search
    lines = output.split('\n')
    result = [m.group(1) for m in map(search, lines) if m]
    return result