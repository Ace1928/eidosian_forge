import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowCredential(command.ShowOne):
    _description = _('Display credential details')

    def get_parser(self, prog_name):
        parser = super(ShowCredential, self).get_parser(prog_name)
        parser.add_argument('credential', metavar='<credential-id>', help=_('ID of credential to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        credential = utils.find_resource(identity_client.credentials, parsed_args.credential)
        credential._info.pop('links')
        return zip(*sorted(credential._info.items()))