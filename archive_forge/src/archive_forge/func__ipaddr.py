from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
@pass_environment
def _ipaddr(*args, **kwargs):
    """This filter is designed to return the input value if a query is True, and False if a query is False"""
    keys = ['value', 'query', 'version', 'alias']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    try:
        if isinstance(data['value'], str):
            pass
        elif isinstance(data['value'], list):
            pass
        elif isinstance(data['value'], int):
            pass
        else:
            raise AnsibleFilterError('Unrecognized type <{0}> for ipaddr filter <{1}>'.format(type(data['value']), 'value'))
    except (TypeError, ValueError):
        raise AnsibleFilterError('Unrecognized type <{0}> for ipaddr filter <{1}>'.format(type(data['value']), 'value'))
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='ipaddr')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return ipaddr(**updated_data)