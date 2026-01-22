from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six
def ParseMetadataArg(metadata=None, resource_type=None):
    """Parses and creates the metadata object from the parsed arguments.

  Args:
    metadata: dict, key-value pairs passed in from the --metadata flag.
    resource_type: string, the type of the resource to be created or updated.

  Returns:
    A message object depending on resource_type.

    Service.MetadataValue message when resource_type='service' and
    Endpoint.MetadataValue message when resource_type='endpoint'.
  """
    if not metadata:
        return None
    msgs = apis.GetMessagesModule(_API_NAME, _VERSION_MAP.get(base.ReleaseTrack.BETA))
    additional_properties = []
    if resource_type == 'endpoint':
        metadata_value_msg = msgs.Endpoint.MetadataValue
    elif resource_type == 'service':
        metadata_value_msg = msgs.Service.MetadataValue
    else:
        return None
    for key, value in six.iteritems(metadata):
        additional_properties.append(metadata_value_msg.AdditionalProperty(key=key, value=value))
    return metadata_value_msg(additionalProperties=additional_properties)