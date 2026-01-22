import logging
from osc_lib.command import command
from openstackclient.i18n import _
def _get_role_ids(identity_client, parsed_args):
    """Return prior and implied role id(s)

    If prior and implied role id(s) are retrievable from identity
    client, return tuple containing them.
    """
    role_id = None
    implied_role_id = None
    roles = identity_client.roles.list()
    for role in roles:
        role_id_or_name = (role.name, role.id)
        if parsed_args.role in role_id_or_name:
            role_id = role.id
        elif parsed_args.implied_role in role_id_or_name:
            implied_role_id = role.id
    return (role_id, implied_role_id)