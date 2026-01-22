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
def _GetCollectionFromMethod(self, api_version, get_method):
    """Created collection_info object given discovery doc get_method."""
    collection_name = _ExtractCollectionName(get_method['id'])
    collection_name = collection_name.split('.', 1)[1]
    flat_path = get_method.get('flatPath')
    path = get_method.get('path')
    return self._MakeResourceCollection(api_version, collection_name, path, flat_path)