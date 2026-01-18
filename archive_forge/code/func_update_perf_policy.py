from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_perf_policy(client_obj, perf_policy_resp, **kwargs):
    if utils.is_null_or_empty(perf_policy_resp):
        return (False, False, 'Update performance policy failed. Performance policy name is not present.', {}, {})
    try:
        perf_policy_name = perf_policy_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(perf_policy_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            perf_policy_resp = client_obj.performance_policies.update(id=perf_policy_resp.attrs.get('id'), **params)
            return (True, True, f"Performance policy '{perf_policy_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, perf_policy_resp.attrs)
        else:
            return (True, False, f"Performance policy '{perf_policy_name}' already present in given state.", {}, perf_policy_resp.attrs)
    except Exception as ex:
        return (False, False, f'Performance policy update failed | {ex}', {}, {})