from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def MergeMetadata(args, api_version):
    """Creates the metadata for the Node.

  Based on googlecloudsdk.command_lib.compute.tpus.tpu_vm.util.MergeMetadata.

  Args:
    args:  The gcloud args
    api_version: The api version (e.g. v2 or v2alpha1)

  Returns:
    The constructed MetadataValue.
  """
    metadata_dict = metadata_utils.ConstructMetadataDict(args.metadata, args.metadata_from_file)
    tpu_messages = GetMessagesModule(api_version)
    metadata = tpu_messages.Node.MetadataValue()
    for key, value in six.iteritems(metadata_dict):
        metadata.additionalProperties.append(tpu_messages.Node.MetadataValue.AdditionalProperty(key=key, value=value))
    return metadata