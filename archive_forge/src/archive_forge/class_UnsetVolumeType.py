import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class UnsetVolumeType(command.Command):
    _description = _('Unset volume type properties')

    def get_parser(self, prog_name):
        parser = super(UnsetVolumeType, self).get_parser(prog_name)
        parser.add_argument('volume_type', metavar='<volume-type>', help=_('Volume type to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', dest='properties', help=_('Remove a property from this volume type (repeat option to remove multiple properties)'))
        parser.add_argument('--project', metavar='<project>', help=_('Removes volume type access to project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--encryption-type', action='store_true', help=_('Remove the encryption type for this volume type (admin only)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        identity_client = self.app.client_manager.identity
        volume_type = utils.find_resource(volume_client.volume_types, parsed_args.volume_type)
        result = 0
        if parsed_args.properties:
            try:
                volume_type.unset_keys(parsed_args.properties)
            except Exception as e:
                LOG.error(_('Failed to unset volume type properties: %s'), e)
                result += 1
        if parsed_args.project:
            project_info = None
            try:
                project_info = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain)
                volume_client.volume_type_access.remove_project_access(volume_type.id, project_info.id)
            except Exception as e:
                LOG.error(_('Failed to remove volume type access from project: %s'), e)
                result += 1
        if parsed_args.encryption_type:
            try:
                volume_client.volume_encryption_types.delete(volume_type)
            except Exception as e:
                LOG.error(_('Failed to remove the encryption type for this volume type: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('Command Failed: One or more of the operations failed'))