from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
def _GetParentCollection(collection_info):
    """Generates the name and path for a parent collection.

  Args:
    collection_info: resource.CollectionInfo, The collection to calculate the
      parent of.

  Returns:
    (str, str), A tuple of parent name and path or (None, None) if there is no
    parent.
  """
    params = collection_info.GetParams(DEFAULT_PATH_NAME)
    if len(params) < 2:
        return (None, None)
    path = collection_info.GetPath(DEFAULT_PATH_NAME)
    parts = path.split('/')
    _PopSegments(parts, True)
    _PopSegments(parts, False)
    if not parts:
        return (None, None)
    parent_path = '/'.join(parts)
    _PopSegments(parts, True)
    if not parts:
        return (None, None)
    if '.' in collection_info.name:
        parent_name, _ = collection_info.name.rsplit('.', 1)
    else:
        parent_name = parts[-1]
    return (parent_name, parent_path)