from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_volcoll(client_obj, volcoll_resp, **kwargs):
    if utils.is_null_or_empty(volcoll_resp):
        return (False, False, 'Update volume collection failed as volume collection is not present.', {}, {})
    try:
        volcoll_name = volcoll_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(volcoll_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            volcoll_resp = client_obj.volume_collections.update(id=volcoll_resp.attrs.get('id'), **params)
            return (True, True, f"Volume collection '{volcoll_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, volcoll_resp.attrs)
        else:
            return (True, False, f"Volume collection '{volcoll_name}' already present in given state.", {}, volcoll_resp.attrs)
    except Exception as ex:
        return (False, False, f'Volume collection update failed | {ex}', {}, {})