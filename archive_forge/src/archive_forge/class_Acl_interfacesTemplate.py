from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
class Acl_interfacesTemplate(NetworkTemplate):

    def __init__(self, lines=None, module=None):
        super(Acl_interfacesTemplate, self).__init__(lines=lines, tmplt=self, module=module)
    PARSERS = [{'name': 'interface', 'getval': re.compile('\n              ^interface\\s\n              (?P<name>\\S+)$', re.VERBOSE), 'setval': 'interface {{ name }}', 'result': {'{{ name }}': {'name': '{{ name }}', 'access_groups': {}}}, 'shared': True}, {'name': 'access_groups', 'getval': re.compile('\n                \\s+(?P<afi>ip|ipv6)\n                (\\saccess-group\\s(?P<acl_name>\\S+))?\n                (\\straffic-filter\\s(?P<acl_name_traffic>\\S+))?\n                \\s(?P<direction>\\S+)$\n                ', re.VERBOSE), 'setval': "{{ 'ip access-group' if afi == 'ipv4' else 'ipv6 traffic-filter' }} {{ name|string }} {{ direction }}", 'result': {'{{ name }}': {'access_groups': {"{{ 'ipv4' if afi == 'ip' else 'ipv6' }}": {'afi': "{{ 'ipv4' if afi == 'ip' else 'ipv6' }}", 'acls': [{'name': '{{ acl_name|string if acl_name is defined else acl_name_traffic }}', 'direction': '{{ direction }}'}]}}}}}]