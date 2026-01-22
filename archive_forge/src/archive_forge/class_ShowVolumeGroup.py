import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowVolumeGroup(command.ShowOne):
    """Show detailed information for a volume group.

    This command requires ``--os-volume-api-version`` 3.13 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Name or ID of volume group.'))
        parser.add_argument('--volumes', action='store_true', dest='show_volumes', default=None, help=_('Show volumes included in the group. (supported by --os-volume-api-version 3.25 or above)'))
        parser.add_argument('--no-volumes', action='store_false', dest='show_volumes', help=_('Do not show volumes included in the group. (supported by --os-volume-api-version 3.25 or above)'))
        parser.add_argument('--replication-targets', action='store_true', dest='show_replication_targets', default=None, help=_('Show replication targets for the group. (supported by --os-volume-api-version 3.38 or above)'))
        parser.add_argument('--no-replication-targets', action='store_false', dest='show_replication_targets', help=_('Do not show replication targets for the group. (supported by --os-volume-api-version 3.38 or above)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.13'):
            msg = _("--os-volume-api-version 3.13 or greater is required to support the 'volume group show' command")
            raise exceptions.CommandError(msg)
        kwargs = {}
        if parsed_args.show_volumes is not None:
            if volume_client.api_version < api_versions.APIVersion('3.25'):
                msg = _("--os-volume-api-version 3.25 or greater is required to support the '--(no-)volumes' option")
                raise exceptions.CommandError(msg)
            kwargs['list_volume'] = parsed_args.show_volumes
        if parsed_args.show_replication_targets is not None:
            if volume_client.api_version < api_versions.APIVersion('3.38'):
                msg = _("--os-volume-api-version 3.38 or greater is required to support the '--(no-)replication-targets' option")
                raise exceptions.CommandError(msg)
        group = utils.find_resource(volume_client.groups, parsed_args.group)
        group = volume_client.groups.show(group.id, **kwargs)
        if parsed_args.show_replication_targets:
            replication_targets = volume_client.groups.list_replication_targets(group.id)
            group.replication_targets = replication_targets
        return _format_group(group)