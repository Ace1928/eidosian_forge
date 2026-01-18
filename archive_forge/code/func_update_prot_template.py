from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_prot_template(client_obj, prot_template_resp, **kwargs):
    if utils.is_null_or_empty(prot_template_resp):
        return (False, False, 'Update protection template failed as protection template is not present.', {}, {})
    try:
        prot_template_name = prot_template_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(prot_template_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            prot_template_resp = client_obj.protection_templates.update(id=prot_template_resp.attrs.get('id'), **params)
            return (True, True, f"Protection template '{prot_template_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, prot_template_resp.attrs)
        else:
            return (True, False, f"Protection template '{prot_template_name}' already present in given state.", {}, prot_template_resp.attrs)
    except Exception as ex:
        return (False, False, f'Protection template update failed | {ex}', {}, {})