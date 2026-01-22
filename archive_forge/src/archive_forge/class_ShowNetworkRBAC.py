import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowNetworkRBAC(command.ShowOne):
    _description = _('Display network RBAC policy details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkRBAC, self).get_parser(prog_name)
        parser.add_argument('rbac_policy', metavar='<rbac-policy>', help=_('RBAC policy (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_rbac_policy(parsed_args.rbac_policy, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)