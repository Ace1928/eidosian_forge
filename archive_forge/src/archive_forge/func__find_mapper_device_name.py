from __future__ import (absolute_import, division, print_function)
import collections
import errno
import glob
import json
import os
import re
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.six import iteritems
from ansible.module_utils.facts import timeout
def _find_mapper_device_name(self, dm_device):
    dm_prefix = '/dev/dm-'
    mapper_device = dm_device
    if dm_device.startswith(dm_prefix):
        dmsetup_cmd = self.module.get_bin_path('dmsetup', True)
        mapper_prefix = '/dev/mapper/'
        rc, dm_name, err = self.module.run_command('%s info -C --noheadings -o name %s' % (dmsetup_cmd, dm_device))
        if rc == 0:
            mapper_device = mapper_prefix + dm_name.rstrip()
    return mapper_device