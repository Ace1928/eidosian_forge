from osc_lib.command import command
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ShareInstanceListExportLocation(command.Lister):
    """List share instance export locations."""
    _description = _('List share instance export locations')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceListExportLocation, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID of the share instance.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        instance = osc_utils.find_resource(share_client.share_instances, parsed_args.instance)
        export_locations = share_client.share_instance_export_locations.list(instance, search_opts=None)
        columns = ['ID', 'Path', 'Is Admin Only', 'Preferred']
        data = (osc_utils.get_dict_properties(export_location._info, columns) for export_location in export_locations)
        return (columns, data)