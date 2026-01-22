from __future__ import (absolute_import, division, print_function)
import os
import re
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
class HPUXVirtualCollector(VirtualCollector):
    _fact_class = HPUXVirtual
    _platform = 'HP-UX'