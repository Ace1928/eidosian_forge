from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.parse_cli_textfsm import (
@pass_environment
def _parse_cli_textfsm(*args, **kwargs):
    """Extend vlan data"""
    keys = ['value', 'template']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='parse_cli_textfsm')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return parse_cli_textfsm(**updated_data)