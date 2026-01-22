from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import fnmatch
import os
import re
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _GetObjectDetails(self, object_path):
    """Gets the actual object data for a given GCS path.

    Args:
      object_path: str, The gs:// path to an object or directory.

    Returns:
      (bool, data), Where element 0 is True if the path is an object, False if
      a directory and where data is either a storage.Object message (for
      objects) or a storage_util.ObjectReference for directories.
    """
    details = self._object_details.get(object_path)
    if details:
        return (True, details)
    else:
        return (False, storage_util.ObjectReference.FromUrl(object_path, allow_empty_object=True))