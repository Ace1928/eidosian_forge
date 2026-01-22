from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.module_utils.common.to_paths import to_paths
def _to_paths(*args, **kwargs):
    """Flatten a complex object into a dictionary of paths and values."""
    keys = ['var', 'prepend', 'wantlist']
    data = dict(zip(keys, args))
    data.update(kwargs)
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='to_paths')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    return to_paths(**updated_data)