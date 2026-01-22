import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreatePolicy(command.ShowOne):
    _description = _('Create new policy')

    def get_parser(self, prog_name):
        parser = super(CreatePolicy, self).get_parser(prog_name)
        parser.add_argument('--type', metavar='<type>', default='application/json', help=_('New MIME type of the policy rules file (defaults to application/json)'))
        parser.add_argument('rules', metavar='<filename>', help=_('New serialized policy rules file'))
        return parser

    def take_action(self, parsed_args):
        blob = utils.read_blob_file_contents(parsed_args.rules)
        identity_client = self.app.client_manager.identity
        policy = identity_client.policies.create(blob=blob, type=parsed_args.type)
        policy._info.pop('links')
        policy._info.update({'rules': policy._info.pop('blob')})
        return zip(*sorted(policy._info.items()))