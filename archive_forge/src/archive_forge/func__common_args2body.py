from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _common_args2body(client, parsed_args, is_create=True):
    if parsed_args.redirect_url:
        if parsed_args.action != 'REDIRECT_TO_URL':
            raise exceptions.CommandError(_('Action must be REDIRECT_TO_URL'))
    if parsed_args.redirect_pool:
        if parsed_args.action != 'REDIRECT_TO_POOL':
            raise exceptions.CommandError(_('Action must be REDIRECT_TO_POOL'))
        parsed_args.redirect_pool_id = _get_pool_id(client, parsed_args.redirect_pool)
    if parsed_args.action == 'REDIRECT_TO_URL' and (not parsed_args.redirect_url):
        raise exceptions.CommandError(_('Redirect URL must be specified'))
    if parsed_args.action == 'REDIRECT_TO_POOL' and (not parsed_args.redirect_pool):
        raise exceptions.CommandError(_('Redirect pool must be specified'))
    attributes = ['name', 'description', 'action', 'redirect_pool_id', 'redirect_url', 'position', 'admin_state_up']
    if is_create:
        parsed_args.listener_id = _get_listener_id(client, parsed_args.listener)
        attributes.extend(['listener_id', 'tenant_id'])
    body = {}
    neutronV20.update_dict(parsed_args, body, attributes)
    return {'l7policy': body}