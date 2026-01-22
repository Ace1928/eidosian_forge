from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __extract_mount_subvolid(self, mount_options):
    for option in mount_options.split(','):
        if option.startswith('subvolid='):
            return int(option[len('subvolid='):])
    raise BtrfsModuleException("Failed to find subvolid for mountpoint in options '%s'" % mount_options)