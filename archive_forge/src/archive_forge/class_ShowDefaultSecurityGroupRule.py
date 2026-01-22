import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class ShowDefaultSecurityGroupRule(command.ShowOne):
    """Show a security group rule used for new default security groups.

    This shows a rule that will be added to any new default security groups
    created. This rule may not be present on existing default security groups.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('rule', metavar='<rule>', help=_('Default security group rule to display (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.sdk_connection.network
        obj = client.find_default_security_group_rule(parsed_args.rule, ignore_missing=False)
        if not obj['remote_ip_prefix']:
            obj['remote_ip_prefix'] = network_utils.format_remote_ip_prefix(obj)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)