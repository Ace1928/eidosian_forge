from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import exceptions
def AddPatchDeploymentsCreateFlags(parser, api_version):
    """Adds flags for os-config create command to parser."""
    parser.add_argument('PATCH_DEPLOYMENT_ID', type=str, help='        Name of the patch deployment to create.\n\n        This name must contain only lowercase letters, numbers, and hyphens,\n        start with a letter, end with a number or a letter, be between 1-63\n        characters, and unique within the project.')
    parser.add_argument('--file', required=True, help='        The JSON or YAML file with the patch deployment to create. For\n        information about the patch deployment format, see https://cloud.google.com/compute/docs/osconfig/rest/{api_version}/projects.patchDeployments.'.format(api_version=api_version))