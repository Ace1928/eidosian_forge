from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def CreateCustomMetadata(entries=None, custom_metadata=None):
    """Creates a custom MetadataValue object.

  Inserts the key/value pairs in entries.

  Args:
    entries: (Dict[str, Any] or None) The dictionary containing key/value pairs
        to insert into metadata. Both the key and value must be able to be
        casted to a string type.
    custom_metadata (apitools_messages.Object.MetadataValue or None): A
        pre-existing custom metadata object to add to. If one is not provided,
        a new one will be constructed.

  Returns:
    An apitools_messages.Object.MetadataValue.
  """
    if custom_metadata is None:
        custom_metadata = apitools_messages.Object.MetadataValue(additionalProperties=[])
    if entries is None:
        entries = {}
    for key, value in six.iteritems(entries):
        custom_metadata.additionalProperties.append(apitools_messages.Object.MetadataValue.AdditionalProperty(key=str(key), value=str(value)))
    return custom_metadata