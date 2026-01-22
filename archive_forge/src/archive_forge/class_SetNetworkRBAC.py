import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetworkRBAC(common.NeutronCommandWithExtraArgs):
    _description = _('Set network RBAC policy properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkRBAC, self).get_parser(prog_name)
        parser.add_argument('rbac_policy', metavar='<rbac-policy>', help=_('RBAC policy to be modified (ID only)'))
        parser.add_argument('--target-project', metavar='<target-project>', help=_('The project to which the RBAC policy will be enforced (name or ID)'))
        parser.add_argument('--target-project-domain', metavar='<target-project-domain>', help=_('Domain the target project belongs to (name or ID). This can be used in case collisions between project names exist.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_rbac_policy(parsed_args.rbac_policy, ignore_missing=False)
        attrs = {}
        if parsed_args.target_project:
            identity_client = self.app.client_manager.identity
            project_id = identity_common.find_project(identity_client, parsed_args.target_project, parsed_args.target_project_domain).id
            attrs['target_tenant'] = project_id
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_rbac_policy(obj, **attrs)