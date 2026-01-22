from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
class DarwinHardwareCollector(HardwareCollector):
    _fact_class = DarwinHardware
    _platform = 'Darwin'