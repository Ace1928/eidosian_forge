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
class BlobKey(object):

    def __init__(self, blob_key):
        """Constructor.

    Used to convert a string to a BlobKey.  Normally used internally by
    Blobstore API.

    Args:
      blob_key:  Key name of BlobReference that this key belongs to.
    """
        ValidateString(blob_key, 'blob-key', empty_ok=True)
        self.__blob_key = blob_key

    def __str__(self):
        """Convert to string."""
        return self.__blob_key

    def __repr__(self):
        """Returns an eval()able string representation of this key.

    Returns a Python string of the form 'datastore_types.BlobKey(...)'
    that can be used to recreate this key.

    Returns:
      string
    """
        return 'datastore_types.%s(%r)' % (type(self).__name__, self.__blob_key)

    def __cmp__(self, other):
        if type(other) is type(self):
            return cmp(str(self), str(other))
        elif isinstance(other, six_subset.string_types):
            return cmp(self.__blob_key, other)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.__blob_key)

    def ToXml(self):
        return str(self)