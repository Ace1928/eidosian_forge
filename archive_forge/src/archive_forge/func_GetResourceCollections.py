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
def GetResourceCollections(self, custom_resources, api_version):
    """Returns all resources collections found in this discovery doc.

    Args:
      custom_resources: {str, str}, A mapping of collection name to path that
          have been registered manually in the yaml file.
      api_version: Override api_version for each found resource collection.

    Returns:
      list(resource_util.CollectionInfo).
    """
    collections = self._ExtractResources(api_version, self._discovery_doc_dict)
    collections.extend(self._GenerateMissingParentCollections(collections, custom_resources, api_version))
    return collections