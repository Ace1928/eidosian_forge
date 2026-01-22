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
def _GenerateMissingParentCollections(self, collections, custom_resources, api_version):
    """Generates parent collections for any existing collection missing one.

    Args:
      collections: [resource.CollectionInfo], The existing collections from the
        discovery doc.
      custom_resources: {str, str}, A mapping of collection name to path that
        have been registered manually in the yaml file.
      api_version: Override api_version for each found resource collection.

    Raises:
      ConflictingCollection: If multiple parent collections have the same name
        but different paths, and a custom resource has not been declared to
        resolve the conflict.

    Returns:
      [resource.CollectionInfo], Additional collections to include in the
      resource module.
    """
    all_names = {c.name: c for c in collections}
    all_paths = {c.GetPath(DEFAULT_PATH_NAME) for c in collections}
    generated = []
    in_progress = list(collections)
    to_process = []
    ignored = {}
    while in_progress:
        for c in in_progress:
            parent_name, parent_path = _GetParentCollection(c)
            if not parent_name:
                continue
            if parent_path in all_paths:
                continue
            if parent_name in custom_resources:
                ignored.setdefault(parent_name, set()).add(parent_path)
                continue
            if parent_name in all_names:
                raise ConflictingCollection('In API [{api}/{version}], the parent of collection [{c}] is not registered, but a collection with [{parent_name}] and path [{existing_path}] already exists. Update the api config file to manually add the parent collection with a path of [{parent_path}].'.format(api=c.api_name, version=api_version, c=c.name, parent_name=parent_name, existing_path=all_names[parent_name].GetPath(DEFAULT_PATH_NAME), parent_path=parent_path))
            parent_collection = self.MakeResourceCollection(parent_name, parent_path, True, api_version)
            to_process.append(parent_collection)
            all_names[parent_name] = parent_collection
            all_paths.add(parent_path)
        generated.extend(to_process)
        in_progress = to_process
        to_process = []
    for name, paths in six.iteritems(ignored):
        if len(paths) > 1:
            continue
        path = paths.pop()
        if path == custom_resources[name]['path']:
            print('WARNING: Custom resource [{}] in API [{}/{}] is redundant.'.format(name, self.api_name, api_version))
    return generated