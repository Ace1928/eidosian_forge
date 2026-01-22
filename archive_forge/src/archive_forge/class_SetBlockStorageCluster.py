from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetBlockStorageCluster(command.Command):
    """Set block storage cluster properties.

    This command requires ``--os-volume-api-version`` 3.7 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('Name of block storage cluster to update (name only)'))
        parser.add_argument('--binary', metavar='<binary>', default='cinder-volume', help=_("Name of binary to filter by; defaults to 'cinder-volume' (optional)"))
        enabled_group = parser.add_mutually_exclusive_group()
        enabled_group.add_argument('--enable', action='store_false', dest='disabled', default=None, help=_('Enable cluster'))
        enabled_group.add_argument('--disable', action='store_true', dest='disabled', help=_('Disable cluster'))
        parser.add_argument('--disable-reason', metavar='<reason>', dest='disabled_reason', help=_('Reason for disabling the cluster (should be used with --disable option)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.7'):
            msg = _("--os-volume-api-version 3.7 or greater is required to support the 'block storage cluster set' command")
            raise exceptions.CommandError(msg)
        if parsed_args.disabled_reason and (not parsed_args.disabled):
            msg = _('Cannot specify --disable-reason without --disable')
            raise exceptions.CommandError(msg)
        cluster = volume_client.clusters.update(parsed_args.cluster, parsed_args.binary, disabled=parsed_args.disabled, disabled_reason=parsed_args.disabled_reason)
        return _format_cluster(cluster, detailed=True)