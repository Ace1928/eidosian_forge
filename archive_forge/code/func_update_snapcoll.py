from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_snapcoll(client_obj, snapcoll_resp, **kwargs):
    if utils.is_null_or_empty(snapcoll_resp):
        return (False, False, 'Update snapshot collection failed as snapshot collection is not present.', {}, {})
    try:
        snapcoll_name = snapcoll_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(snapcoll_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            snapcoll_resp = client_obj.snapshot_collections.update(id=snapcoll_resp.attrs.get('id'), **params)
            return (True, True, f"Snapshot collection '{snapcoll_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, snapcoll_resp.attrs)
        else:
            return (True, False, f"Snapshot collection '{snapcoll_name}' already present in given state.", {}, snapcoll_resp.attrs)
    except Exception as ex:
        return (False, False, f'Snapshot collection update failed | {ex}', {}, {})