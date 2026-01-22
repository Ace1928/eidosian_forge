from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospfv3_area_bfd_multiplier(config_data):
    if 'bfd' in config_data:
        if 'multiplier' in config_data['bfd']:
            command = 'area {area_id} '.format(**config_data)
            command += 'bfd multiplier {multiplier}'.format(**config_data['bfd'])
            return command