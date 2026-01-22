from osc_lib.command import command
from osc_lib import utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShareSnapshotInstanceExportLocationList(command.Lister):
    """List export locations from a share snapshot instance."""
    _description = _('List export locations from a share snapshot instance.')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotInstanceExportLocationList, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('Name or ID of the share instance.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_instance = apiutils.find_resource(share_client.share_snapshot_instances, parsed_args.instance)
        share_snapshot_instance_export_locations = share_client.share_snapshot_instance_export_locations.list(snapshot_instance=snapshot_instance)
        columns = ['ID', 'Path', 'Is Admin only']
        return (columns, (utils.get_item_properties(s, columns) for s in share_snapshot_instance_export_locations))