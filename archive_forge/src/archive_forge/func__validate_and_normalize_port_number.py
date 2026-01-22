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
def _validate_and_normalize_port_number(self, acl_type, acl_name, rule, endpoint):
    port_number = rule.get(endpoint, {}).get('port_number')
    if port_number:
        if port_number.get('gt') == L4_PORT_START:
            port_number['lt'] = L4_PORT_END
            del port_number['gt']
        elif rule[endpoint]['port_number'].get('range'):
            port_range = rule[endpoint]['port_number']['range']
            if port_range['begin'] >= port_range['end']:
                self._invalid_rule('begin must be less than end in {0} -> port_number -> range'.format(endpoint), acl_type, acl_name, rule['sequence_num'])
            if port_range['begin'] == L4_PORT_START:
                port_number['lt'] = port_range['end']
                del port_number['range']
            elif port_range['end'] == L4_PORT_END:
                port_number['gt'] = port_range['begin']
                del port_number['range']