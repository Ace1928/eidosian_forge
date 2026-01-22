from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
@staticmethod
def _normalize_protocol_options(rule):
    tcp = rule.get('protocol_options', {}).get('tcp')
    if tcp:
        if not any(list(tcp.values())):
            del rule['protocol_options']
        else:
            tcp_flag_list = list(tcp.keys())
            for tcp_flag in tcp_flag_list:
                if not tcp[tcp_flag]:
                    del tcp[tcp_flag]