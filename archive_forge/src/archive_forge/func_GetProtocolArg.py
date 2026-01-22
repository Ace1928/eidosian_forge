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
def GetProtocolArg(messages):
    """Creates a --protocol flag spec for the arg parser.

  Args:
    messages: The messages module.

  Returns:
    The chosen protocol arg.
  """
    protocol_arg = arg_utils.ChoiceEnumMapper('--protocol', messages.Instance.ProtocolValueValuesEnum, help_str='The service protocol for the Cloud Filestore instance.', custom_mappings={'NFS_V3': ('nfs-v3', 'NFSv3 protocol.'), 'NFS_V4_1': ('nfs-v4-1', 'NFSv4.1 protocol.')}, default='NFS_V3')
    return protocol_arg