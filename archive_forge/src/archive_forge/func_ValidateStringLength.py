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
def ValidateStringLength(name, value, max_len):
    """Raises an exception if the supplied string is too long.

  Args:
    name: Name of the property this is for.
    value: String value.
    max_len: Maximum length the string may be.

  Raises:
    OverflowError if the value is larger than the maximum length.
  """
    if isinstance(value, six_subset.text_type):
        value = value.encode('utf-8')
    if len(value) > max_len:
        raise datastore_errors.BadValueError('Property %s is %d bytes long; it must be %d or less. Consider Text instead, which can store strings of any length.' % (name, len(value), max_len))