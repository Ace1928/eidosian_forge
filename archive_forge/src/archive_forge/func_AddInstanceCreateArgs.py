from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddInstanceCreateArgs(parser, api_version):
    """Add args for creating an instance."""
    concept_parsers.ConceptParser([flags.GetInstancePresentationSpec('The instance to create.')]).AddToParser(parser)
    AddDescriptionArg(parser)
    AddLocationArg(parser)
    AddRegionArg(parser)
    AddAsyncFlag(parser)
    labels_util.AddCreateLabelsFlags(parser)
    AddNetworkArg(parser)
    messages = filestore_client.GetMessages(version=api_version)
    GetTierArg(messages).choice_arg.AddToParser(parser)
    if api_version == filestore_client.BETA_API_VERSION:
        GetProtocolArg(messages).choice_arg.AddToParser(parser)
        AddConnectManagedActiveDirectoryArg(parser)
    AddFileShareArg(parser, api_version, include_snapshot_flags=api_version == filestore_client.ALPHA_API_VERSION, include_backup_flags=True)
    if api_version in [filestore_client.BETA_API_VERSION, filestore_client.V1_API_VERSION]:
        AddKmsKeyArg(parser)