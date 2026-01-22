import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class FailoverVolumeGroup(command.Command):
    """Failover replication for a volume group.

    This command requires ``--os-volume-api-version`` 3.38 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Name or ID of volume group to failover replication for.'))
        parser.add_argument('--allow-attached-volume', action='store_true', dest='allow_attached_volume', default=False, help=_('Allow group with attached volumes to be failed over.'))
        parser.add_argument('--disallow-attached-volume', action='store_false', dest='allow_attached_volume', default=False, help=_('Disallow group with attached volumes to be failed over.'))
        parser.add_argument('--secondary-backend-id', metavar='<backend_id>', help=_('Secondary backend ID.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.38'):
            msg = _("--os-volume-api-version 3.38 or greater is required to support the 'volume group failover' command")
            raise exceptions.CommandError(msg)
        group = utils.find_resource(volume_client.groups, parsed_args.group)
        volume_client.groups.failover_replication(group.id, allow_attached_volume=parsed_args.allow_attached_volume, secondary_backend_id=parsed_args.secondary_backend_id)