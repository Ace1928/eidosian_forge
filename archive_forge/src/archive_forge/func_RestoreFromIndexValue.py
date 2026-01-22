from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
def RestoreFromIndexValue(index_value, data_type):
    """Restores a index value to the correct datastore type.

  Projection queries return property values direclty from a datastore index.
  These values are the native datastore values, one of str, bool, long, float,
  GeoPt, Key or User. This function restores the original value when the
  original type is known.

  This function returns the value type returned when decoding a normal entity,
  not necessarily of type data_type. For example, data_type=int returns a
  long instance.

  Args:
    index_value: The value returned by FromPropertyPb for the projected
      property.
    data_type: The type of the value originally given to ToPropertyPb

  Returns:
    The restored property value.

  Raises:
    datastore_errors.BadValueError if the value cannot be restored.
  """
    raw_type = _PROPERTY_TYPE_TO_INDEX_VALUE_TYPE.get(data_type)
    if raw_type is None:
        raise datastore_errors.BadValueError('Unsupported data type (%r)' % data_type)
    if index_value is None:
        return index_value
    if not isinstance(index_value, raw_type):
        raise datastore_errors.BadValueError('Unsupported converstion. Expected %r got %r' % (type(index_value), raw_type))
    meaning = _PROPERTY_MEANINGS.get(data_type)
    if isinstance(index_value, str) and meaning not in _NON_UTF8_MEANINGS:
        index_value = six_subset.text_type(index_value, 'utf-8')
    conv = _PROPERTY_CONVERSIONS.get(meaning)
    if not conv:
        return index_value
    try:
        value = conv(index_value)
    except (KeyError, ValueError, IndexError, TypeError, AttributeError) as msg:
        raise datastore_errors.BadValueError('Error converting value: %r\nException was: %s' % (index_value, msg))
    return value