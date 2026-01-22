import datetime
import logging
from keystoneclient import exceptions as identity_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowTrust(command.ShowOne):
    _description = _('Display trust details')

    def get_parser(self, prog_name):
        parser = super(ShowTrust, self).get_parser(prog_name)
        parser.add_argument('trust', metavar='<trust>', help=_('Trust to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        trust = utils.find_resource(identity_client.trusts, parsed_args.trust)
        trust._info.pop('roles_links', None)
        trust._info.pop('links', None)
        roles = trust._info.pop('roles')
        msg = ' '.join((r['name'] for r in roles))
        trust._info['roles'] = msg
        return zip(*sorted(trust._info.items()))