from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_default_information(config_data):
    if 'default_information' in config_data:
        command = 'default-information'
        if 'originate' in config_data['default_information']:
            command += ' originate'
        if 'always' in config_data['default_information']:
            command += ' always'
        if 'metric' in config_data['default_information']:
            command += ' metric {metric}'.format(**config_data['default_information'])
        if 'metric_type' in config_data['default_information']:
            command += ' metric-type {metric_type}'.format(**config_data['default_information'])
        if 'metric' in config_data['default_information']:
            command += ' route-map {route_map}'.format(**config_data['default_information'])
        return command