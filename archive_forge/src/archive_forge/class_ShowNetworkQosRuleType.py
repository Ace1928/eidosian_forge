from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowNetworkQosRuleType(command.ShowOne):
    _description = _('Show details about supported QoS rule type')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkQosRuleType, self).get_parser(prog_name)
        parser.add_argument('rule_type', metavar='<qos-rule-type-name>', help=_('Name of QoS rule type'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.get_qos_rule_type(parsed_args.rule_type)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)