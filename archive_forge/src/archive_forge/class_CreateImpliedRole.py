import logging
from osc_lib.command import command
from openstackclient.i18n import _
class CreateImpliedRole(command.ShowOne):
    _description = _('Creates an association between prior and implied roles')

    def get_parser(self, prog_name):
        parser = super(CreateImpliedRole, self).get_parser(prog_name)
        parser.add_argument('role', metavar='<role>', help=_('Role (name or ID) that implies another role'))
        parser.add_argument('--implied-role', metavar='<role>', help='<role> (name or ID) implied by another role', required=True)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        prior_role_id, implied_role_id = _get_role_ids(identity_client, parsed_args)
        response = identity_client.inference_rules.create(prior_role_id, implied_role_id)
        response._info.pop('links', None)
        return zip(*sorted([(k, v['id']) for k, v in response._info.items()]))