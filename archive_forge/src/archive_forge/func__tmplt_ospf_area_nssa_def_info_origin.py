from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_nssa_def_info_origin(config_data):
    if 'nssa' in config_data:
        command = 'area {area_id} nssa'.format(**config_data)
        if 'default_information_originate' in config_data['nssa']:
            command += ' default-information-originate'
            def_info_origin = config_data['nssa'].get('default_information_originate')
            if 'metric' in def_info_origin:
                command += ' metric {metric}'.format(**config_data['nssa']['default_information_originate'])
            if 'metric_type' in def_info_origin:
                command += ' metric-type {metric_type}'.format(**config_data['nssa']['default_information_originate'])
        return command