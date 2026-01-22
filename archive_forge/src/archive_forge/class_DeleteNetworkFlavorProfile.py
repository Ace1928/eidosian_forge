import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteNetworkFlavorProfile(command.Command):
    _description = _('Delete network flavor profile')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkFlavorProfile, self).get_parser(prog_name)
        parser.add_argument('flavor_profile', metavar='<flavor-profile>', nargs='+', help=_('Flavor profile(s) to delete (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for flavor_profile in parsed_args.flavor_profile:
            try:
                obj = client.find_service_profile(flavor_profile, ignore_missing=False)
                client.delete_service_profile(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete flavor profile with ID '%(flavor_profile)s': %(e)s"), {'flavor_profile': flavor_profile, 'e': e})
        if result > 0:
            total = len(parsed_args.flavor_profile)
            msg = _('%(result)s of %(total)s flavor_profiles failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)