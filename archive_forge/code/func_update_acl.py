from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def update_acl(consul_client, configuration):
    """
    Updates an ACL.
    :param consul_client: the consul client
    :param configuration: the run configuration
    :return: the output of the update
    """
    existing_acl = load_acl_with_token(consul_client, configuration.token)
    changed = existing_acl.rules != configuration.rules
    if changed:
        name = configuration.name if configuration.name is not None else existing_acl.name
        rules_as_hcl = encode_rules_as_hcl_string(configuration.rules)
        updated_token = consul_client.acl.update(configuration.token, name=name, type=configuration.token_type, rules=rules_as_hcl)
        if updated_token != configuration.token:
            raise AssertionError()
    return Output(changed=changed, token=configuration.token, rules=configuration.rules, operation=UPDATE_OPERATION)