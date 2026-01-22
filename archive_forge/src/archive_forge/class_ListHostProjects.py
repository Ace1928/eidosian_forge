from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import xpn_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.xpn import util as command_lib_util
from googlecloudsdk.command_lib.organizations import flags as organizations_flags
from googlecloudsdk.core import properties
class ListHostProjects(base.ListCommand):
    """List shared VPC host projects in a given organization."""
    detailed_help = {'EXAMPLES': '\n          To list the shared VPC host projects in the organization with ID\n          12345, run:\n\n            $ {command} 12345\n\n            NAME       CREATION_TIMESTAMP            XPN_PROJECT_STATUS\n            xpn-host1  2000-01-01T12:00:00.00-00:00  HOST\n            xpn-host2  2000-01-02T12:00:00.00-00:00  HOST\n      '}

    @staticmethod
    def Args(parser):
        organizations_flags.IdArg('whose XPN host projects to list.').AddToParser(parser)
        parser.display_info.AddFormat(command_lib_util.XPN_PROJECTS_FORMAT)

    def Run(self, args):
        xpn_client = xpn_api.GetXpnClient(self.ReleaseTrack())
        project = properties.VALUES.core.project.Get(required=True)
        organization_id = args.id
        return xpn_client.ListOrganizationHostProjects(project, organization_id=organization_id)