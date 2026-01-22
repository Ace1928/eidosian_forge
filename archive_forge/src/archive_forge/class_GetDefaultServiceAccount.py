from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.core import properties
class GetDefaultServiceAccount(base.Command):
    """Get the default service account for a project."""
    detailed_help = {'DESCRIPTION': 'Get the default service account for a project.', 'EXAMPLES': '\n            To get the default service account for the project:\n\n                $ {command}\n            '}

    @staticmethod
    def Args(parser):
        flags.AddRegionFlag(parser)

    def Run(self, args):
        serviceaccount_region = args.region or cloudbuild_util.DEFAULT_REGION
        client = cloudbuild_util.GetClientInstance()
        return client.projects_locations.GetDefaultServiceAccount(client.MESSAGES_MODULE.CloudbuildProjectsLocationsGetDefaultServiceAccountRequest(name='projects/%s/locations/%s/defaultServiceAccount' % (properties.VALUES.core.project.GetOrFail(), serviceaccount_region)))