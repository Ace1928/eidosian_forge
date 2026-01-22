from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import ConfigBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.community.network.plugins.module_utils.network.exos.facts.facts import Facts
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import send_requests
import json
from copy import deepcopy
def _update_lldp_config_body_if_diff(self, want, have, request):
    if want.get('interval'):
        if want['interval'] != have['interval']:
            request['data']['openconfig-lldp:config'].update({'hello-timer': want['interval']})
    if want.get('tlv_select'):
        want_suppress = [key.upper() for key, value in want['tlv_select'].items() if have['tlv_select'][key] != value and value is False]
        if want_suppress:
            want_suppress.extend([key.upper() for key, value in have['tlv_select'].items() if value is False])
            request['data']['openconfig-lldp:config'].update({'suppress-tlv-advertisement': want_suppress})
            request['data']['openconfig-lldp:config']['suppress-tlv-advertisement'].sort()