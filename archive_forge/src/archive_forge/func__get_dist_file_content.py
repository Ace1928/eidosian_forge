from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def _get_dist_file_content(self, path, allow_empty=False):
    if not _file_exists(path, allow_empty=allow_empty):
        return (False, None)
    data = self._get_file_content(path)
    return (True, data)