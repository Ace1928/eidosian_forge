from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_log_adjacency_changes(proc):
    command = 'log-adjacency-changes'
    if proc.get('log_adjacency_changes').get('detail', False) is True:
        command += ' detail'
    return command