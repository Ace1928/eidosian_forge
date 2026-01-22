from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetStoragePoolServiceLevelArg(messages, required=True):
    """Adds a --service-level arg to the given parser.

  Args:
    messages: The messages module.
    required: bool, whether choice arg is required or not

  Returns:
    the choice arg.
  """
    custom_mappings = {'PREMIUM': ('premium', '\n                          Premium Service Level for Cloud NetApp Storage Pool.\n                          The Premium Service Level has a throughput per TiB of\n                          allocated volume size of 64 MiB/s.'), 'EXTREME': ('extreme', '\n                          Extreme Service Level for Cloud NetApp Storage Pool.\n                          The Extreme Service Level has a throughput per TiB of\n                          allocated volume size of 128 MiB/s.'), 'STANDARD': ('standard', '\n                          Standard Service Level for Cloud NetApp Storage Pool.\n                          The Standard Service Level has a throughput per TiB of\n                          allocated volume size of 128 MiB/s.')}
    service_level_arg = arg_utils.ChoiceEnumMapper('--service-level', messages.StoragePool.ServiceLevelValueValuesEnum, help_str='The service level for the Cloud NetApp Storage Pool.\n       For more details, see:\n       https://cloud.google.com/netapp/volumes/docs/configure-and-use/storage-pools/overview#service_levels\n        ', custom_mappings=custom_mappings, required=required)
    return service_level_arg