from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ListIpOwners(base.ListCommand):
    """List IP owners with optional filters in a network."""
    example = '\n    List all IP owners in a network:\n\n      $ {command} my-network\n\n    List IP owners only for given owner types:\n\n      $ {command} my-network           --owner-types=instance,address,forwardingRule\n\n    List IP owners only for given owner projects:\n\n      $ {command} my-network           --owner-projects=p1,p2\n\n    List IP owners only for given subnet:\n\n      $ {command} my-network           --subnet-name=subnet-1 --subnet-region=us-central1\n\n    List IP owners whose IP ranges overlap with the given IP CIDR range:\n\n      $ {command} my-network           --ip-cidr-range=10.128.10.130/30\n  '
    detailed_help = {'brief': 'List IP owners in a network.', 'DESCRIPTION': '*{command}* is used to list IP owners in a network.', 'EXAMPLES': example}

    @staticmethod
    def Args(parser):
        parser.add_argument('name', help='The name of the network.')
        parser.add_argument('--subnet-name', help='Only return IP owners in the subnets with the name. Not applicable for legacy networks.')
        parser.add_argument('--subnet-region', help='Only return IP owners in the subnets of the region. Not applicable for legacy networks.')
        parser.add_argument('--ip-cidr-range', help='Only return IP owners whose IP ranges overlap with the IP CIDR range.')
        parser.add_argument('--owner-projects', help='Only return IP owners in the projects. Multiple projects are separated by comma, e.g., "project-1,project-2".')
        parser.add_argument('--owner-types', help='Only return IP owners of the types, which can be any combination of instance, address, forwardingRule, subnetwork, or network. Multiple types are separated by comma, e.g., "instance,forwardingRule,address"')
        parser.display_info.AddFormat("\n        table(\n            ipCidrRange:label=IP_CIDR_RANGE,\n            systemOwned:label=SYSTEM_OWNED,\n            owners.join(','):label=OWNERS)\n    ")

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client.apitools_client
        messages = client.MESSAGES_MODULE
        project = properties.VALUES.core.project.Get(required=True)
        request = messages.ComputeNetworksListIpOwnersRequest(project=project, network=args.name, subnetName=args.subnet_name, subnetRegion=args.subnet_region, ipCidrRange=args.ip_cidr_range, ownerProjects=args.owner_projects, ownerTypes=args.owner_types)
        items = list_pager.YieldFromList(client.networks, request, method='ListIpOwners', field='items', limit=args.limit, batch_size=None)
        return items