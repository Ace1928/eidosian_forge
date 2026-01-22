import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetProject(command.Command):
    _description = _('Set project properties')

    def get_parser(self, prog_name):
        parser = super(SetProject, self).get_parser(prog_name)
        parser.add_argument('project', metavar='<project>', help=_('Project to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set project name'))
        parser.add_argument('--description', metavar='<description>', help=_('Set project description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable project'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable project'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a project property (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        project = utils.find_resource(identity_client.tenants, parsed_args.project)
        kwargs = project._info
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if parsed_args.enable:
            kwargs['enabled'] = True
        if parsed_args.disable:
            kwargs['enabled'] = False
        if parsed_args.property:
            kwargs.update(parsed_args.property)
        if 'id' in kwargs:
            del kwargs['id']
        if 'name' in kwargs:
            kwargs['tenant_name'] = kwargs['name']
            del kwargs['name']
        identity_client.tenants.update(project.id, **kwargs)