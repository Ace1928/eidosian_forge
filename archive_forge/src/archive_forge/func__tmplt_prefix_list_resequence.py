from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_prefix_list_resequence(config_data):
    command = 'resequence'
    config_data = config_data['prefix_lists'].get('entries', {})
    for k, v in iteritems(config_data):
        if v['resequence'].get('start_seq'):
            command += ' ' + str(v['resequence']['start_seq'])
        if v['resequence'].get('step'):
            command += ' ' + str(v['resequence']['step'])
    return command