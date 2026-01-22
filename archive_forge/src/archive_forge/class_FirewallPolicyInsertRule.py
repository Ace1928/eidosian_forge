import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class FirewallPolicyInsertRule(neutronv20.UpdateCommand):
    """Insert a rule into a given firewall policy."""
    resource = 'firewall_policy'

    def call_api(self, neutron_client, firewall_policy_id, body):
        return neutron_client.firewall_policy_insert_rule(firewall_policy_id, body)

    def args2body(self, parsed_args):
        _rule = ''
        if parsed_args.firewall_rule_id:
            _rule = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'firewall_rule', parsed_args.firewall_rule_id)
        _insert_before = ''
        if 'insert_before' in parsed_args:
            if parsed_args.insert_before:
                _insert_before = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'firewall_rule', parsed_args.insert_before)
        _insert_after = ''
        if 'insert_after' in parsed_args:
            if parsed_args.insert_after:
                _insert_after = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'firewall_rule', parsed_args.insert_after)
        body = {'firewall_rule_id': _rule, 'insert_before': _insert_before, 'insert_after': _insert_after}
        return body

    def get_parser(self, prog_name):
        parser = super(FirewallPolicyInsertRule, self).get_parser(prog_name)
        parser.add_argument('--insert-before', metavar='FIREWALL_RULE', help=_('Insert before this rule.'))
        parser.add_argument('--insert-after', metavar='FIREWALL_RULE', help=_('Insert after this rule.'))
        parser.add_argument('firewall_rule_id', metavar='FIREWALL_RULE', help=_('New rule to insert.'))
        self.add_known_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        body = self.args2body(parsed_args)
        _id = neutronv20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.id)
        self.call_api(neutron_client, _id, body)
        print(_('Inserted firewall rule in firewall policy %(id)s') % {'id': parsed_args.id}, file=self.app.stdout)