from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddMigrateRuntimeFlags(api_version, parser):
    """Construct groups and arguments specific to the runtime migration."""
    post_startup_script_option_choices = ['POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED', 'POST_STARTUP_SCRIPT_OPTION_SKIP', 'POST_STARTUP_SCRIPT_OPTION_RERUN']
    AddRuntimeResource(api_version, parser)
    network_group = parser.add_group(help='Network configs.')
    AddNetworkArgument('The name of the VPC that this instance is in. Format: projects/`{project_id}`/global/networks/`{network_id}`.', network_group)
    AddSubnetArgument('The name of the subnet that this instance is in. Format: projects/`{project_id}`/regions/`{region}`/subnetworks/`{subnetwork_id}`.', network_group)
    parser.add_argument('--service-account', help='The service account to be included in the Compute Engine instance of the new Workbench Instance when the Runtime uses single user only mode for permission. If not specified, the [Compute Engine default service account](https://cloud.google.com/compute/docs/access/service-accounts#default_service_account) is used. When the Runtime uses service account mode for permission, it will reuse the same service account, and this field must be empty.')
    parser.add_argument('--post-startup-script-option', help='Specifies the behavior of post startup script during migration.', choices=post_startup_script_option_choices, default='POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED')