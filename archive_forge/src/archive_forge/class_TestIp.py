from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_firewall_api_client as api_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import firewall_rules_util
from googlecloudsdk.core import log
class TestIp(base.Command):
    """Display firewall rules that match a given IP."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To test an IP address against the firewall rule set, run:\n\n              $ {command} 127.1.2.3\n          '}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat(firewall_rules_util.LIST_FORMAT)
        parser.add_argument('ip', help='An IPv4 or IPv6 address to test against the firewall.')

    def Run(self, args):
        client = api_client.GetApiClientForTrack(self.ReleaseTrack())
        response = client.List(args.ip)
        rules = list(response)
        if rules:
            log.status.Print('The action `{0}` will apply to the IP address.\n'.format(rules[0].action))
            log.status.Print('Matching Rules')
        else:
            log.status.Print('No rules match the IP address.')
        return rules