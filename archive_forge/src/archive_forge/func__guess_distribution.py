from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def _guess_distribution(self):
    dist = (get_distribution(), get_distribution_version(), get_distribution_codename())
    distribution_guess = {'distribution': dist[0] or 'NA', 'distribution_version': dist[1] or 'NA', 'distribution_release': 'NA' if dist[2] is None else dist[2]}
    distribution_guess['distribution_major_version'] = distribution_guess['distribution_version'].split('.')[0] or 'NA'
    return distribution_guess