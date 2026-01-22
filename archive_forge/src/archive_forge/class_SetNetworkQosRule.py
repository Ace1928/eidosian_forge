import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class SetNetworkQosRule(common.NeutronCommandWithExtraArgs):
    _description = _('Set Network QoS rule properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkQosRule, self).get_parser(prog_name)
        parser.add_argument('qos_policy', metavar='<qos-policy>', help=_('QoS policy that contains the rule (name or ID)'))
        parser.add_argument('id', metavar='<rule-id>', help=_('Network QoS rule to delete (ID)'))
        _add_rule_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            qos = network_client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False)
            rule_type = _find_rule_type(qos, parsed_args.id)
            if not rule_type:
                raise Exception('Rule not found')
            attrs = _get_attrs(network_client, parsed_args)
            attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
            qos_id = attrs.pop('qos_policy_id')
            qos_rule = _rule_action_call(network_client, ACTION_FIND, rule_type)(attrs.pop('id'), qos_id)
            _rule_action_call(network_client, ACTION_SET, rule_type)(qos_rule, qos_id, **attrs)
        except Exception as e:
            msg = _('Failed to set Network QoS rule ID "%(rule)s": %(e)s') % {'rule': parsed_args.id, 'e': e}
            raise exceptions.CommandError(msg)