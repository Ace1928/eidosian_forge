import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetRole(command.Command):
    _description = _('Set role properties')

    def get_parser(self, prog_name):
        parser = super(SetRole, self).get_parser(prog_name)
        parser.add_argument('role', metavar='<role>', help=_('Role to modify (name or ID)'))
        parser.add_argument('--description', metavar='<description>', help=_('Add description about the role'))
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain the role belongs to (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set role name'))
        common.add_resource_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        domain_id = None
        if parsed_args.domain:
            domain_id = common.find_domain(identity_client, parsed_args.domain).id
        options = common.get_immutable_options(parsed_args)
        role = utils.find_resource(identity_client.roles, parsed_args.role, domain_id=domain_id)
        identity_client.roles.update(role.id, name=parsed_args.name, description=parsed_args.description, options=options)