from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def validate_merge_group(client_obj, group_name, **kwargs):
    if utils.is_null_or_empty(group_name):
        return (False, False, 'Validate merge for group failed as it is not present.', {}, {})
    try:
        group_resp = client_obj.groups.get(id=None, name=group_name)
        if utils.is_null_or_empty(group_resp):
            return (False, False, f"Validate merge for group '{group_name}' cannot be done as it is not present.", {}, {})
        params = utils.remove_null_args(**kwargs)
        validate_merge_resp = client_obj.groups.validate_merge(id=group_resp.attrs.get('id'), **params)
        if hasattr(validate_merge_resp, 'attrs'):
            validate_merge_resp = validate_merge_resp.attrs
        if utils.is_null_or_empty(validate_merge_resp.get('validation_error_msg')):
            return (True, False, f"Validate merge operation for group '{group_name}' done successfully.", {}, validate_merge_resp)
        else:
            msg = validate_merge_resp.get('validation_error_msg')
            return (False, False, f"Validate merge operation for group '{group_name}' failed with error '{msg}'", {}, validate_merge_resp)
    except Exception as ex:
        return (False, False, f"Validate merge for group failed | '{ex}'", {}, {})