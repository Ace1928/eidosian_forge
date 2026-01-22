from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_event_log(config_data):
    if 'event_log' in config_data:
        command = 'event-log'
        if 'one_shot' in config_data['event_log']:
            command += ' one-shot'
        if 'pause' in config_data['event_log']:
            command += ' pause'
        if 'size' in config_data['event_log']:
            command += ' size {size}'.format(**config_data['event_log'])
        return command