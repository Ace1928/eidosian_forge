from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetEffectiveFirewalls(base.DescribeCommand, base.ListCommand):
    """Get the effective firewalls for a Compute Engine virtual machine network interface.

  ## EXAMPLES

  To get the effective firewalls for an instance, run:

    $ {command} example-instance --zone us-central1-a

  gets the effective firewalls applied on the default network interface of a
  Compute Engine virtual machine "example-instance" in zone
  us-central1-a
  """
    _support_network_firewall_policy = True

    @staticmethod
    def Args(parser):
        instances_flags.INSTANCE_ARG.AddArgument(parser)
        parser.add_argument('--network-interface', default='nic0', help='The name of the network interface to get the effective firewalls for.')
        parser.display_info.AddFormat(firewalls_utils.EFFECTIVE_FIREWALL_LIST_FORMAT)
        lister.AddBaseListerArgs(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        messages = holder.client.messages
        instance_ref = instances_flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetDefaultScopeLister(holder.client))
        instance = client.apitools_client.instances.Get(messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))
        for i in instance.networkInterfaces:
            if i.name == args.network_interface:
                break
        else:
            raise exceptions.UnknownArgumentException('network-interface', 'Instance does not have a network interface [{}], present interfaces are [{}].'.format(args.network_interface, ', '.join([i.name for i in instance.networkInterfaces])))
        request = messages.ComputeInstancesGetEffectiveFirewallsRequest(project=instance_ref.project, instance=instance_ref.instance, zone=instance_ref.zone, networkInterface=args.network_interface)
        res = client.apitools_client.instances.GetEffectiveFirewalls(request)
        org_firewall = []
        network_firewall = []
        all_firewall_policy = []
        if hasattr(res, 'firewalls'):
            network_firewall = firewalls_utils.SortNetworkFirewallRules(client, res.firewalls)
        if hasattr(res, 'firewallPolicys') and res.firewallPolicys:
            for fp in res.firewallPolicys:
                firewall_policy_rule = firewalls_utils.SortFirewallPolicyRules(client, fp.rules)
                fp_response = client.messages.InstancesGetEffectiveFirewallsResponseEffectiveFirewallPolicy(name=fp.name, rules=firewall_policy_rule, type=fp.type)
                all_firewall_policy.append(fp_response)
        elif hasattr(res, 'organizationFirewalls'):
            for sp in res.organizationFirewalls:
                org_firewall_rule = firewalls_utils.SortOrgFirewallRules(client, sp.rules)
                org_firewall.append(client.messages.InstancesGetEffectiveFirewallsResponseOrganizationFirewallPolicy(id=sp.id, rules=org_firewall_rule))
        if args.IsSpecified('format') and args.format == 'json':
            if org_firewall:
                return client.messages.InstancesGetEffectiveFirewallsResponse(organizationFirewalls=org_firewall, firewalls=network_firewall, firewallPolicys=all_firewall_policy)
            else:
                return client.messages.InstancesGetEffectiveFirewallsResponse(firewalls=network_firewall, firewallPolicys=all_firewall_policy)
        result = []
        for fp in all_firewall_policy:
            result.extend(firewalls_utils.ConvertFirewallPolicyRulesToEffectiveFwRules(client, fp, self._support_network_firewall_policy))
        for sp in org_firewall:
            result.extend(firewalls_utils.ConvertOrgSecurityPolicyRulesToEffectiveFwRules(sp))
        result.extend(firewalls_utils.ConvertNetworkFirewallRulesToEffectiveFwRules(network_firewall))
        return result

    def Epilog(self, resources_were_displayed):
        del resources_were_displayed
        log.status.Print('\n' + firewalls_utils.LIST_NOTICE)