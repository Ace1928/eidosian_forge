from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_partner(client_obj, downstream_hostname, secret, **kwargs):
    if utils.is_null_or_empty(downstream_hostname):
        return (False, False, 'Update replication partner failed as no downstream partner is provided.', {}, {})
    try:
        upstream_repl_resp = client_obj.replication_partners.get(id=None, hostname=downstream_hostname)
        if utils.is_null_or_empty(upstream_repl_resp):
            return (False, False, f"Replication partner '{downstream_hostname}' cannot be updated as it is not present.", {}, {})
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(upstream_repl_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            upstream_repl_resp = client_obj.replication_partners.update(id=upstream_repl_resp.attrs.get('id'), secret=secret, **params)
            return (True, True, f"Replication partner '{downstream_hostname}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, upstream_repl_resp.attrs)
        else:
            return (True, False, f"Replication partner '{upstream_repl_resp.attrs.get('name')}' already present in given state.", {}, upstream_repl_resp.attrs)
    except Exception as ex:
        return (False, False, f'Replication partner update failed |{ex}', {}, {})