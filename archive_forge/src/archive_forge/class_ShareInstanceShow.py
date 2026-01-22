import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
class ShareInstanceShow(command.ShowOne):
    """Show share instance."""
    _description = _('Show share instance')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceShow, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID of the share instance.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        instance = osc_utils.find_resource(share_client.share_instances, parsed_args.instance)
        export_locations = share_client.share_instance_export_locations.list(instance)
        instance._info['export_locations'] = []
        for export_location in export_locations:
            export_location._info.pop('links', None)
            instance._info['export_locations'].append(export_location._info)
        if parsed_args.formatter == 'table':
            instance._info['export_locations'] = cliutils.convert_dict_list_to_string(instance._info['export_locations'])
        instance._info.pop('links', None)
        return self.dict2columns(instance._info)