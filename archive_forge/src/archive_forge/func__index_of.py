from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.index_of import index_of
@pass_environment
def _index_of(*args, **kwargs):
    """Find the indicies of items in a list matching some criteria."""
    keys = ['environment', 'data', 'test', 'value', 'key', 'fail_on_missing', 'wantlist']
    data = dict(zip(keys, args))
    data.update(kwargs)
    environment = data.pop('environment')
    aav = AnsibleArgSpecValidator(data=data, schema=DOCUMENTATION, name='index_of')
    valid, errors, updated_data = aav.validate()
    if not valid:
        raise AnsibleFilterError(errors)
    updated_data['tests'] = environment.tests
    return index_of(**updated_data)