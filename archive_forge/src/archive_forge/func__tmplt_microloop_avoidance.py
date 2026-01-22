from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_microloop_avoidance(config_data):
    if 'microloop_avoidance' in config_data:
        command = 'microloop avoidance'
        if 'protected' in config_data['microloop_avoidance']:
            command += ' protected'
        if 'segment_routing' in config_data['microloop_avoidance']:
            command += ' segment_routing'
        if 'rib_update_delay' in config_data['microloop_avoidance']:
            command += ' rin-update-delay {0}'.config_data['microloop_avoidance'].get('rib_update_delay')
        return command