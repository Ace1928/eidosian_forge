from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _MakeCollectionDict(collection_name):
    """Returns a dictionary of collection attributes from Registry.

  Args:
    collection_name: Name of collection to create dictionary about.
  """
    collection_info = resources.REGISTRY.GetCollectionInfo(collection_name)
    collection_dict = {}
    collection_dict['collection_name'] = collection_name
    collection_dict['api_name'] = collection_info.api_name
    collection_dict['uppercase_api_name'] = collection_info.api_name.capitalize()
    flat_paths = collection_info.flat_paths
    collection_dict['use_relative_name'] = 'false' if not flat_paths else 'true'
    collection_dict['api_version'] = collection_info.api_version
    collection_dict['release_tracks'] = _GetReleaseTracks(collection_info.api_version)
    collection_dict['plural_resource_name'] = collection_info.name.split('.')[-1]
    collection_dict['singular_name'] = _MakeSingular(collection_dict['plural_resource_name'])
    collection_dict['flags'] = ' '.join(['--' + param + '=my-' + param for param in collection_info.params if param not in (collection_dict['singular_name'], 'project')])
    collection_dict['collection_name'] = collection_name
    collection_dict['parent'] = 'location' if 'location' in collection_name else 'project'
    return collection_dict