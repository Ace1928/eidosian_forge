import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetVolumeGroup(command.ShowOne):
    """Update a volume group.

    This command requires ``--os-volume-api-version`` 3.13 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Name or ID of volume group.'))
        parser.add_argument('--name', metavar='<name>', help=_('New name for group.'))
        parser.add_argument('--description', metavar='<description>', help=_('New description for group.'))
        parser.add_argument('--enable-replication', action='store_true', dest='enable_replication', default=None, help=_('Enable replication for group. (supported by --os-volume-api-version 3.38 or above)'))
        parser.add_argument('--disable-replication', action='store_false', dest='enable_replication', help=_('Disable replication for group. (supported by --os-volume-api-version 3.38 or above)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.13'):
            msg = _("--os-volume-api-version 3.13 or greater is required to support the 'volume group set' command")
            raise exceptions.CommandError(msg)
        group = utils.find_resource(volume_client.groups, parsed_args.group)
        if parsed_args.enable_replication is not None:
            if volume_client.api_version < api_versions.APIVersion('3.38'):
                msg = _("--os-volume-api-version 3.38 or greater is required to support the '--enable-replication' or '--disable-replication' options")
                raise exceptions.CommandError(msg)
            if parsed_args.enable_replication:
                volume_client.groups.enable_replication(group.id)
            else:
                volume_client.groups.disable_replication(group.id)
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['description'] = parsed_args.description
        if kwargs:
            group = volume_client.groups.update(group.id, **kwargs)
        return _format_group(group)