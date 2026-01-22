from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __parse_filesystem(self, line):
    label = re.sub('\\s*uuid:.*$', '', re.sub('^Label:\\s*', '', line))
    id = re.sub('^.*uuid:\\s*', '', line)
    filesystem = {}
    filesystem['label'] = label.strip("'") if label != 'none' else None
    filesystem['uuid'] = id
    filesystem['devices'] = []
    filesystem['mountpoints'] = []
    filesystem['subvolumes'] = []
    filesystem['default_subvolid'] = None
    return filesystem