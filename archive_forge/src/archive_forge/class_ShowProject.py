import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowProject(command.ShowOne):
    _description = _('Display project details')

    def get_parser(self, prog_name):
        parser = super(ShowProject, self).get_parser(prog_name)
        parser.add_argument('project', metavar='<project>', help=_('Project to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        info = {}
        try:
            project = utils.find_resource(identity_client.tenants, parsed_args.project)
            info.update(project._info)
        except ks_exc.Forbidden:
            auth_ref = self.app.client_manager.auth_ref
            if parsed_args.project == auth_ref.project_id or parsed_args.project == auth_ref.project_name:
                info = {'id': auth_ref.project_id, 'name': auth_ref.project_name, 'enabled': True}
            else:
                raise
        info.pop('parent_id', None)
        reserved = ('name', 'id', 'enabled', 'description')
        properties = {}
        for k, v in list(info.items()):
            if k not in reserved:
                info.pop(k)
                if v is not None:
                    properties[k] = v
        info['properties'] = format_columns.DictColumn(properties)
        return zip(*sorted(info.items()))