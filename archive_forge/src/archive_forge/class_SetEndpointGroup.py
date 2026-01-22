from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
class SetEndpointGroup(command.Command):
    _description = _('Set endpoint group properties')

    def get_parser(self, prog_name):
        parser = super(SetEndpointGroup, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument('--name', metavar='<name>', help=_('Set a name for the endpoint group'))
        parser.add_argument('endpoint_group', metavar='<endpoint-group>', help=_('Endpoint group to set (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        if parsed_args.name:
            attrs['name'] = str(parsed_args.name)
        endpoint_id = client.find_vpn_endpoint_group(parsed_args.endpoint_group, ignore_missing=False)['id']
        try:
            client.update_vpn_endpoint_group(endpoint_id, **attrs)
        except Exception as e:
            msg = _('Failed to set endpoint group %(endpoint_group)s: %(e)s') % {'endpoint_group': parsed_args.endpoint_group, 'e': e}
            raise exceptions.CommandError(msg)