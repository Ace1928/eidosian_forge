import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
class UnsetFirewallPolicy(command.Command):
    _description = _('Unset firewall policy properties')

    def get_parser(self, prog_name):
        parser = super(UnsetFirewallPolicy, self).get_parser(prog_name)
        parser.add_argument(const.FWP, metavar='<firewall-policy>', help=_('Firewall policy to unset (name or ID)'))
        firewall_rule_group = parser.add_mutually_exclusive_group()
        firewall_rule_group.add_argument('--firewall-rule', action='append', metavar='<firewall-rule>', help=_('Remove firewall rule(s) from the firewall policy (name or ID)'))
        firewall_rule_group.add_argument('--all-firewall-rule', action='store_true', help=_('Remove all firewall rules from the firewall policy'))
        parser.add_argument('--audited', action='store_true', help=_('Disable auditing for the policy'))
        parser.add_argument('--share', action='store_true', help=_('Restrict use of the firewall policy to the current project'))
        return parser

    def _get_attrs(self, client_manager, parsed_args):
        attrs = {}
        client = client_manager.network
        if parsed_args.firewall_rule:
            current = client.find_firewall_policy(parsed_args.firewall_policy)[const.FWRS]
            removed = []
            for f in set(parsed_args.firewall_rule):
                removed.append(client.find_firewall_rule(f)['id'])
            attrs[const.FWRS] = [r for r in current if r not in removed]
        if parsed_args.all_firewall_rule:
            attrs[const.FWRS] = []
        if parsed_args.audited:
            attrs['audited'] = False
        if parsed_args.share:
            attrs['shared'] = False
        return attrs

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fwp_id = client.find_firewall_policy(parsed_args.firewall_policy)['id']
        attrs = self._get_attrs(self.app.client_manager, parsed_args)
        try:
            client.update_firewall_policy(fwp_id, **attrs)
        except Exception as e:
            msg = _("Failed to unset firewall policy '%(policy)s': %(e)s") % {'policy': parsed_args.firewall_policy, 'e': e}
            raise exceptions.CommandError(msg)