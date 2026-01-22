from __future__ import (absolute_import, division, print_function)
import os
import re
import time
from ansible.module_utils.six.moves import reduce
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.timeout import TimeoutError, timeout
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.facts.sysctl import get_sysctl
class NetBSDHardwareCollector(HardwareCollector):
    _fact_class = NetBSDHardware
    _platform = 'NetBSD'