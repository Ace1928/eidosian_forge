from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags as shared_flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.persistent_resources import persistent_resource_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCreatePersistentResourceFlags(parser):
    """Adds flags related to create a Persistent Resource."""
    shared_flags.AddRegionResourceArg(parser, 'to create a Persistent Resource', prompt_func=region_util.GetPromptForRegionFunc(constants.SUPPORTED_TRAINING_REGIONS))
    shared_flags.NETWORK.AddToParser(parser)
    ENABLE_CUSTOM_SERVICE_ACCOUNT.AddToParser(parser)
    shared_flags.AddKmsKeyResourceArg(parser, 'persistent resource')
    labels_util.AddCreateLabelsFlags(parser)
    shared_flags.GetDisplayNameArg('Persistent Resource', required=False).AddToParser(parser)
    resource_id_flag = base.Argument('--persistent-resource-id', required=True, default=None, help='User-specified ID of the Persistent Resource.')
    resource_id_flag.AddToParser(parser)
    resource_pool_spec_group = base.ArgumentGroup(help='resource pool specification.', required=True)
    resource_pool_spec_group.AddArgument(_PERSISTENT_RESOURCE_CONFIG)
    resource_pool_spec_group.AddArgument(_RESOURCE_POOL_SPEC)
    resource_pool_spec_group.AddToParser(parser)