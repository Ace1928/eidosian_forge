from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __get_v1_integer_value(self, v1_value):
    """Returns an integer value from a v1 Value.

    Args:
      v1_value: a googledatastore.Value

    Returns:
      an integer

    Raises:
      InvalidConversionError: if the value doesn't contain an integer value
    """
    check_conversion(v1_value.HasField('integer_value'), 'Value does not contain an integer value.')
    return v1_value.integer_value