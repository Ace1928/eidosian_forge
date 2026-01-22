from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_mount_size
class AIXHardwareCollector(HardwareCollector):
    _platform = 'AIX'
    _fact_class = AIXHardware