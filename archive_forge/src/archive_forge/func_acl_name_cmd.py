from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def acl_name_cmd(self, name, afi, acl_type):
    """generate parent acl command"""
    if afi == 'ipv4':
        if not acl_type:
            try:
                acl_id = int(name)
                if not acl_type:
                    if acl_id >= 1 and acl_id <= 99:
                        acl_type = 'standard'
                    if acl_id >= 100 and acl_id <= 199:
                        acl_type = 'extended'
            except ValueError:
                acl_type = 'extended'
        command = 'ip access-list {0} {1}'.format(acl_type, name)
    elif afi == 'ipv6':
        command = 'ipv6 access-list {0}'.format(name)
    return command