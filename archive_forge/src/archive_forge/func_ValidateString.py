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
def ValidateString(value, name='unused', exception=datastore_errors.BadValueError, max_len=_MAX_STRING_LENGTH, empty_ok=False):
    """Raises an exception if value is not a valid string or a subclass thereof.

  A string is valid if it's not empty, no more than _MAX_STRING_LENGTH bytes,
  and not a Blob. The exception type can be specified with the exception
  argument; it defaults to BadValueError.

  Args:
    value: the value to validate.
    name: the name of this value; used in the exception message.
    exception: the type of exception to raise.
    max_len: the maximum allowed length, in bytes.
    empty_ok: allow empty value.
  """
    if value is None and empty_ok:
        return
    if not isinstance(value, basestring) or isinstance(value, Blob):
        raise exception('%s should be a string; received %s (a %s):' % (name, value, typename(value)))
    if not value and (not empty_ok):
        raise exception('%s must not be empty.' % name)
    if len(value.encode('utf-8')) > max_len:
        raise exception('%s must be under %d bytes.' % (name, max_len))