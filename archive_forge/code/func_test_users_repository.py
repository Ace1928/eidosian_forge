from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def test_users_repository(module, repository_id, disable_fail=False):
    """
    Find and return users repository information
    Use disable_fail when we are looking for an user repository
    and it may or may not exist and neither case is an error.
    """
    system = get_system(module)
    name = module.params['name']
    try:
        path = f'config/ldap/{repository_id}/test'
        result = system.api.post(path=path)
    except APICommandFailed as err:
        if disable_fail:
            return False
        msg = f'Users repository {name} testing failed: {str(err)}'
        module.fail_json(msg=msg)
    if result.response.status_code in [200]:
        return True
    return False