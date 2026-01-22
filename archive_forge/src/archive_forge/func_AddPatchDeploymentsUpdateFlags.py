from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import exceptions
def AddPatchDeploymentsUpdateFlags(parser, api_version, release_track):
    """Adds flags for os-config update command to parser."""
    parser.add_argument('PATCH_DEPLOYMENT_ID', type=str, help='        Name of the patch deployment to update.\n\n        To get a list of patch deployments that are available for update, run\n        the `gcloud {release_track} compute os-config patch-deployments list`\n        command.'.format(release_track=release_track))
    parser.add_argument('--file', required=True, help='        The JSON or YAML file with the patch deployment to update. For\n        information about the patch deployment format, see https://cloud.google.com/compute/docs/osconfig/rest/{api_version}/projects.patchDeployments.'.format(api_version=api_version))