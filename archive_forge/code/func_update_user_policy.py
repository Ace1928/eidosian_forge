from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_user_policy(client_obj, **kwargs):
    try:
        user_resp = client_obj.user_policies.get()
        if utils.is_null_or_empty(user_resp):
            return (False, False, 'User policy is not present on Array', {}, {})
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(user_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            user_resp = client_obj.user_policies.update(id=user_resp.attrs.get('id'), **params)
            return (True, True, f"Updated user policy successfully with following attributes '{changed_attrs_dict}'.", changed_attrs_dict, user_resp.attrs)
        else:
            return (True, False, 'User Policy already present in given state.', {}, user_resp.attrs)
    except Exception as ex:
        return (False, False, f'User Policy Update failed | {ex}', {}, {})