import copy
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as fwaas_const
class ShowNetworkLog(command.ShowOne):
    _description = _('Display network log details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkLog, self).get_parser(prog_name)
        parser.add_argument('network_log', metavar='<network-log>', help=_('Network log to show (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.neutronclient
        log_id = client.find_resource('log', parsed_args.network_log, cmd_resource=NET_LOG)['id']
        obj = client.show_network_log(log_id)['log']
        columns, display_columns = column_util.get_columns(obj, _attr_map)
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)