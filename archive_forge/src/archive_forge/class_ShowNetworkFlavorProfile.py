import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowNetworkFlavorProfile(command.ShowOne):
    _description = _('Display network flavor profile details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkFlavorProfile, self).get_parser(prog_name)
        parser.add_argument('flavor_profile', metavar='<flavor-profile>', help=_('Flavor profile to display (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_service_profile(parsed_args.flavor_profile, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)