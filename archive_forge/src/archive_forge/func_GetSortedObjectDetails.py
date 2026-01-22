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
def GetSortedObjectDetails(self, object_paths):
    """Gets all the details for the given paths and returns them sorted.

    Args:
      object_paths: [str], A list of gs:// object or directory paths.

    Returns:
      [{path, data}], A list of dicts with the keys path and data. Path is the
      gs:// path to the object or directory. Object paths will not end in a '/'
      and directory paths will. The data is either a storage.Object message (for
      objects) or a storage_util.ObjectReference for directories. The sort
      order is alphabetical with all directories first and then all objects.
    """
    all_data = []
    for path in object_paths:
        is_obj, data = self._GetObjectDetails(path)
        path = path if is_obj else path + '/'
        all_data.append((is_obj, {'path': path, 'data': data}))
    all_data = sorted(all_data, key=lambda o: (o[0], o[1]['path']))
    return [d[1] for d in all_data]