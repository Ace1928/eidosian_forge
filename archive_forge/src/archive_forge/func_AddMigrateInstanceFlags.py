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
def AddMigrateInstanceFlags(api_version, parser):
    """Construct groups and arguments specific to the instance migration."""
    post_startup_script_option_choices = ['POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED', 'POST_STARTUP_SCRIPT_OPTION_SKIP', 'POST_STARTUP_SCRIPT_OPTION_RERUN']
    AddInstanceResource(api_version, parser)
    parser.add_argument('--post-startup-script-option', help='// Specifies the behavior of post startup script during migration.', choices=post_startup_script_option_choices, default='POST_STARTUP_SCRIPT_OPTION_UNSPECIFIED')