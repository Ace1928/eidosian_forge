from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_set_metric(data):
    cmd = 'set metric'
    metric = data['set']['metric']
    for x in ['bandwidth', 'igrp_delay_metric', 'igrp_reliability_metric', 'igrp_effective_bandwidth_metric', 'igrp_mtu']:
        if x in metric:
            cmd += ' {0}'.format(metric[x])
    return cmd