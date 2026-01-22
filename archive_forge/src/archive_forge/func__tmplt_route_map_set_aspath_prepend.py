from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_set_aspath_prepend(config_data):
    el = config_data['entries']
    command = 'set as-path prepend '
    c = el['set']['as_path']['prepend']
    if c.get('last_as'):
        command += 'last-as ' + str(c['last_as'])
    if c.get('as_number'):
        num = ' '.join(c['as_number'].split(','))
        command += num
    return command