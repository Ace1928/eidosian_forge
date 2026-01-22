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
def _ExtractResources(self, api_version, infos):
    """Extract resource definitions from discovery doc."""
    collections = []
    if infos.get('methods'):
        methods = infos.get('methods')
        get_method = methods.get('get')
        if get_method:
            collection_info = self._GetCollectionFromMethod(api_version, get_method)
            collections.append(collection_info)
    if infos.get('resources'):
        for _, info in infos.get('resources').items():
            subresource_collections = self._ExtractResources(api_version, info)
            collections.extend(subresource_collections)
    return collections