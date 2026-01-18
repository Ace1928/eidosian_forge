from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def validate_volcoll(client_obj, volcoll_name):
    if utils.is_null_or_empty(volcoll_name):
        return (False, False, 'Validate volume collection failed as volume collection name is null.', {}, {})
    try:
        volcoll_resp = client_obj.volume_collections.get(id=None, name=volcoll_name)
        if utils.is_null_or_empty(volcoll_resp):
            return (False, False, f"Volume collection '{volcoll_name}' not present for validation.", {}, {})
        else:
            volcoll_validate_resp = client_obj.volume_collections.validate(id=volcoll_resp.attrs.get('id'))
            if hasattr(volcoll_validate_resp, 'attrs'):
                volcoll_validate_resp = volcoll_validate_resp.attrs
            return (True, False, f"Validation of volume collection '{volcoll_name}' done successfully.", {}, volcoll_validate_resp)
    except Exception as ex:
        return (False, False, f'Validation of volume collection failed | {ex}', {}, {})