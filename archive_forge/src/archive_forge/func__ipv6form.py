from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@pass_environment
def _ipv6form(*args, **kwargs):
    """Convert the given data from json to xml."""
    keys = ['value', 'format']
    data = dict(zip(keys, args[1:]))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='ipv6form')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return ipv6form(**updated_data)