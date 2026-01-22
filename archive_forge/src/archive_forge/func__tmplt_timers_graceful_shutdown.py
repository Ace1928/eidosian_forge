from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_timers_graceful_shutdown(config_data):
    if 'timers' in config_data:
        command = 'timers graceful-shutdown'
        if 'initial_delay' in config_data['timers']['graceful-shutdown']:
            command += ' initial delay {initial_delay}'.format(**config_data['timers']['graceful-shutdown'])
        if 'retain_routes' in config_data['timers']['graceful-shutdown']:
            command += ' retain routes {retain_routes}'.format(**config_data['timers']['graceful-shutdown'])
        return command