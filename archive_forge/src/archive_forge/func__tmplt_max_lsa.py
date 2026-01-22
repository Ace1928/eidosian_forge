from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_max_lsa(proc):
    max_lsa = proc['max_lsa']
    command = 'max-lsa {max_non_self_generated_lsa}'.format(**max_lsa)
    if max_lsa.get('threshold'):
        command += ' {threshold}'.format(**max_lsa)
    if max_lsa.get('warning_only'):
        command += ' warning-only'
    if max_lsa.get('ignore_time'):
        command += ' ignore-time {ignore_time}'.format(**max_lsa)
    if max_lsa.get('ignore_count'):
        command += ' ignore-count {ignore_count}'.format(**max_lsa)
    if max_lsa.get('reset_time'):
        command += ' reset-time {reset_time}'.format(**max_lsa)
    return command