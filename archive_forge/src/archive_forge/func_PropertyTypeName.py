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
def PropertyTypeName(value):
    """Returns the name of the type of the given property value, as a string.

  Raises BadValueError if the value is not a valid property type.

  Args:
    value: any valid property value

  Returns:
    string
  """
    if value.__class__ in _PROPERTY_MEANINGS:
        meaning = _PROPERTY_MEANINGS[value.__class__]
        name = entity_pb.Property._Meaning_NAMES[meaning]
        return name.lower().replace('_', ':')
    elif isinstance(value, basestring):
        return 'string'
    elif isinstance(value, users.User):
        return 'user'
    elif isinstance(value, _PREFERRED_NUM_TYPE):
        return 'int'
    elif value is None:
        return 'null'
    else:
        return typename(value).lower()