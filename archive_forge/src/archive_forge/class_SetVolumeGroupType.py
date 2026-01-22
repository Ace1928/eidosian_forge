import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetVolumeGroupType(command.ShowOne):
    """Update a volume group type.

    This command requires ``--os-volume-api-version`` 3.11 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group_type', metavar='<group_type>', help=_('Name or ID of volume group type.'))
        parser.add_argument('--name', metavar='<name>', help=_('New name for volume group type.'))
        parser.add_argument('--description', metavar='<description>', help=_('New description for volume group type.'))
        type_group = parser.add_mutually_exclusive_group()
        type_group.add_argument('--public', dest='is_public', action='store_true', default=None, help=_('Make volume group type available to other projects.'))
        type_group.add_argument('--private', dest='is_public', action='store_false', help=_('Make volume group type unavailable to other projects.'))
        parser.add_argument('--no-property', action='store_true', help=_('Remove all properties from this volume group type (specify both --no-property and --property to remove the current properties before setting new properties)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Property to add or modify for this volume group type (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.11'):
            msg = _("--os-volume-api-version 3.11 or greater is required to support the 'volume group type set' command")
            raise exceptions.CommandError(msg)
        group_type = utils.find_resource(volume_client.group_types, parsed_args.group_type)
        kwargs = {}
        errors = 0
        if parsed_args.name is not None:
            kwargs['name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['description'] = parsed_args.description
        if parsed_args.is_public is not None:
            kwargs['is_public'] = parsed_args.is_public
        if kwargs:
            try:
                group_type = volume_client.group_types.update(group_type.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to update group type: %s'), e)
                errors += 1
        if parsed_args.no_property:
            try:
                keys = group_type.get_keys().keys()
                group_type.unset_keys(keys)
            except Exception as e:
                LOG.error(_('Failed to clear group type properties: %s'), e)
                errors += 1
        if parsed_args.properties:
            try:
                group_type.set_keys(parsed_args.properties)
            except Exception as e:
                LOG.error(_('Failed to set group type properties: %s'), e)
                errors += 1
        if errors > 0:
            msg = _('Command Failed: One or more of the operations failed')
            raise exceptions.CommandError()
        return _format_group_type(group_type)