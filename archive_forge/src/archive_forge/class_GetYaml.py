from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util
from googlecloudsdk.command_lib.container.fleet.memberships import errors
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class GetYaml(base.Command):
    """Generates YAML for anthos support RBAC policies.

  ## EXAMPLES

    To generate the YAML for support access RBAC policies with membership
    `my-membership`, run:

      $ {command} my-membership

  """

    def GetAnthosSupportUser(self, project_id):
        """Gets P4SA account name for Anthos Support when user not specified.

    Args:
      project_id: the project ID of the resource.

    Returns:
      The P4SA account name for Anthos Support.
    """
        project_number = projects_api.Get(projects_util.ParseProject(project_id)).projectNumber
        hub_endpoint_override = util.APIEndpoint()
        if hub_endpoint_override == util.PROD_API:
            return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='')
        elif hub_endpoint_override == util.STAGING_API:
            return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='staging-')
        elif hub_endpoint_override == util.AUTOPUSH_API:
            return ANTHOS_SUPPORT_USER.format(project_number=project_number, instance_name='autopush-')
        else:
            raise errors.UnknownApiEndpointOverrideError('gkehub')

    @classmethod
    def Args(cls, parser):
        resources.AddMembershipResourceArg(parser, membership_help=textwrap.dedent('            The membership name that you want to generate support access RBAC\n            policies for.\n        '), location_help=textwrap.dedent('            The location of the membership resource, e.g. `us-central1`.\n            If not specified, defaults to `global`.\n        '), membership_required=True, positional=True)
        parser.add_argument('--rbac-output-file', type=str, help=textwrap.dedent('            If specified, the generated RBAC policy will be written to the\n            designated local file.\n        '))

    def Run(self, args):
        project_id = properties.VALUES.core.project.GetOrFail()
        user = self.GetAnthosSupportUser(project_id)
        name = RESOURCE_NAME_FORMAT.format(membership_name=resources.ParseMembershipArg(args), rbacrolebinding_id=ROLE_BINDING_ID)
        fleet_client = client.FleetClient(release_track=self.ReleaseTrack())
        response = fleet_client.GenerateMembershipRbacRoleBindingYaml(name, ROLE_TYPE, user, None)
        if args.rbac_output_file:
            sys.stdout.write('Generated RBAC policy is written to file: {} \n'.format(args.rbac_output_file))
        else:
            sys.stdout.write('Generated RBAC policy is: \n')
            sys.stdout.write('--------------------------------------------\n')
        log.WriteToFileOrStdout(args.rbac_output_file if args.rbac_output_file else '-', response.roleBindingsYaml, overwrite=True, binary=False, private=True)