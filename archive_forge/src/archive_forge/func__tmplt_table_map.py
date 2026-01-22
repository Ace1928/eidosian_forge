from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_table_map(proc):
    table_map = proc['table_map']
    command = 'table-map'
    if table_map.get('name'):
        command += ' {name}'.format(**table_map)
    if table_map.get('filter'):
        command += ' filter'
    return command