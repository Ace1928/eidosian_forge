from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_default_metric(config_data):
    command = 'default-metric {default_metric}'.format(**config_data)
    return command