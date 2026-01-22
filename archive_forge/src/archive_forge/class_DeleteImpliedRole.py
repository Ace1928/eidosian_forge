import logging
from osc_lib.command import command
from openstackclient.i18n import _
class DeleteImpliedRole(command.Command):
    _description = _('Deletes an association between prior and implied roles')

    def get_parser(self, prog_name):
        parser = super(DeleteImpliedRole, self).get_parser(prog_name)
        parser.add_argument('role', metavar='<role>', help=_('Role (name or ID) that implies another role'))
        parser.add_argument('--implied-role', metavar='<role>', help='<role> (name or ID) implied by another role', required=True)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        prior_role_id, implied_role_id = _get_role_ids(identity_client, parsed_args)
        identity_client.inference_rules.delete(prior_role_id, implied_role_id)